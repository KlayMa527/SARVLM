#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SARVLM 图文检索评估脚本

评估指标：
- R@1, R@5, R@10: 检索召回率
- Mean Recall: 平均召回率

支持两种检索任务：
- Image-to-Text (I2T): 用图像检索文本
- Text-to-Image (T2I): 用文本检索图像

Usage:
    python eval_retrieval.py \
        --model coca_ViT-L-14 \
        --checkpoint /path/to/checkpoint.pt \
        --data-csv /path/to/data.csv \
        --batch-size 32 \
        --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# 添加src目录到路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import open_clip
from open_clip import create_model_and_transforms, get_tokenizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="SARVLM 图文检索评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型相关参数
    parser.add_argument(
        "--model", 
        type=str, 
        default="coca_ViT-L-14",
        help="模型架构名称 (如: coca_ViT-L-14, ViT-B-32, ViT-L-14)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="模型权重文件路径 (.pt 或 .bin)"
    )
    parser.add_argument(
        "--pretrained", 
        type=str, 
        default=None,
        help="预训练权重标签 (如果不使用checkpoint)"
    )
    
    # 数据相关参数
    parser.add_argument(
        "--data-csv", 
        type=str, 
        default=" path/to/val_uni_image_and_caption.csv",
        help="评估数据集CSV文件路径"
    )
    parser.add_argument(
        "--img-key", 
        type=str, 
        default="imgpath",
        help="CSV中图像路径列名"
    )
    parser.add_argument(
        "--caption-key", 
        type=str, 
        default="caption",
        help="CSV中文本描述列名"
    )
    
    # 运行相关参数
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="批处理大小"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--precision", 
        type=str, 
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="模型精度"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=4,
        help="数据加载线程数"
    )
    
    # 输出相关参数
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="结果保存目录 (默认为checkpoint同级目录)"
    )
    parser.add_argument(
        "--output-name", 
        type=str, 
        default=None,
        help="输出结果文件名 (不含扩展名，默认为 retrieval_results_{checkpoint_name})"
    )
    parser.add_argument(
        "--save-features", 
        action="store_true",
        help="是否保存提取的特征"
    )
    
    return parser.parse_args()


def load_model(args):
    """加载模型和预处理器"""
    print(f"\n{'='*60}")
    print(f"正在加载模型: {args.model}")
    print(f"{'='*60}")
    
    # 检查是否是TorchScript格式
    is_torchscript = False
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            # 尝试加载并检测是否是TorchScript
            test_load = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            is_torchscript = isinstance(test_load, torch.jit.ScriptModule)
            del test_load
        except Exception as e:
            print(f"预检测权重格式时出错: {e}")
            is_torchscript = False
    
    # 如果是TorchScript格式，使用pretrained参数直接加载
    if is_torchscript and args.checkpoint:
        print(f"检测到TorchScript格式权重")
        print(f"正在加载权重: {args.checkpoint}")
        
        try:
            model, _, preprocess = create_model_and_transforms(
                model_name=args.model,
                pretrained=args.checkpoint,
                precision=args.precision,
                device=args.device,
            )
            print("权重加载完成!")
        except Exception as e:
            print(f"TorchScript加载失败: {e}")
            print("尝试使用普通方式加载...")
            is_torchscript = False
    
    if not is_torchscript:
        # 创建模型（不加载预训练权重）
        model, _, preprocess = create_model_and_transforms(
            model_name=args.model,
            pretrained=args.pretrained,
            precision=args.precision,
            device=args.device,
        )
        
        # 加载checkpoint权重
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"正在加载权重: {args.checkpoint}")
            try:
                # 首先尝试 weights_only=True
                checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
            except Exception as e:
                # 如果失败，使用 weights_only=False
                print(f"使用安全模式加载失败，切换到非安全模式...")
                checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
            
            # 处理TorchScript模型
            if isinstance(checkpoint, torch.jit.ScriptModule):
                print("检测到TorchScript模型，提取state_dict...")
                state_dict = checkpoint.state_dict()
                keys_to_remove = ["input_resolution", "context_length", "vocab_size"]
                for key in keys_to_remove:
                    state_dict.pop(key, None)
            elif isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            if hasattr(state_dict, 'keys'):
                if any(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"警告: 缺少的keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"警告: 多余的keys: {len(unexpected_keys)}")
                print("权重加载完成!")
            else:
                print(f"警告: 无法解析权重格式")
        elif args.checkpoint:
            print(f"警告: checkpoint文件不存在: {args.checkpoint}")
    
    model.eval()
    
    # 加载tokenizer
    tokenizer = get_tokenizer(args.model)
    
    return model, preprocess, tokenizer


def load_data(args):
    """加载评估数据"""
    print(f"\n{'='*60}")
    print(f"正在加载数据: {args.data_csv}")
    print(f"{'='*60}")
    
    df = pd.read_csv(args.data_csv)
    
    print(f"数据量: {len(df)}")
    print(f"图像列: {args.img_key}")
    print(f"文本列: {args.caption_key}")
    
    images = df[args.img_key].tolist()
    captions = df[args.caption_key].tolist()
    
    return images, captions


@torch.no_grad()
def extract_features(model, preprocess, tokenizer, images, captions, args):
    """提取图像和文本特征"""
    print(f"\n{'='*60}")
    print("正在提取特征...")
    print(f"{'='*60}")
    
    device = args.device
    batch_size = args.batch_size
    
    # 存储特征
    image_features_list = []
    text_features_list = []
    
    # 批量处理图像
    print("提取图像特征...")
    for i in tqdm(range(0, len(images), batch_size), desc="图像特征"):
        batch_images = images[i:i + batch_size]
        
        # 加载和预处理图像
        batch_tensors = []
        for img_path in batch_images:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(img)
                batch_tensors.append(img_tensor)
            except Exception as e:
                print(f"警告: 无法加载图像 {img_path}: {e}")
                # 使用空白图像替代
                img_tensor = torch.zeros(3, 224, 224)
                batch_tensors.append(img_tensor)
        
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # 提取特征
        features = model.encode_image(batch_tensor, normalize=True)
        image_features_list.append(features.cpu())
    
    # 批量处理文本
    print("提取文本特征...")
    for i in tqdm(range(0, len(captions), batch_size), desc="文本特征"):
        batch_captions = captions[i:i + batch_size]
        
        # tokenize文本
        text_tokens = tokenizer(batch_captions).to(device)
        
        # 提取特征
        features = model.encode_text(text_tokens, normalize=True)
        text_features_list.append(features.cpu())
    
    # 合并特征
    image_features = torch.cat(image_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)
    
    print(f"图像特征维度: {image_features.shape}")
    print(f"文本特征维度: {text_features.shape}")
    
    return image_features, text_features


def compute_recall_at_k(similarity_matrix, k, mode="i2t"):
    """
    计算Recall@K
    
    Args:
        similarity_matrix: 相似度矩阵 [N_images x N_texts]
        k: top-k
        mode: "i2t" (Image-to-Text) 或 "t2i" (Text-to-Image)
    
    Returns:
        recall: Recall@K值
    """
    n = similarity_matrix.shape[0]
    
    if mode == "i2t":
        # Image-to-Text: 对每个图像，找到最相似的k个文本
        # Ground truth: 第i个图像对应第i个文本
        _, topk_indices = similarity_matrix.topk(k, dim=1)
        correct = torch.arange(n).unsqueeze(1).expand_as(topk_indices)
        hits = (topk_indices == correct).any(dim=1).float()
    else:
        # Text-to-Image: 对每个文本，找到最相似的k个图像
        # Ground truth: 第i个文本对应第i个图像
        sim_t2i = similarity_matrix.T  # [N_texts x N_images]
        _, topk_indices = sim_t2i.topk(k, dim=1)
        correct = torch.arange(n).unsqueeze(1).expand_as(topk_indices)
        hits = (topk_indices == correct).any(dim=1).float()
    
    recall = hits.mean().item() * 100.0
    return recall


def evaluate_retrieval(image_features, text_features):
    """
    评估检索性能
    
    Args:
        image_features: 图像特征 [N x D]
        text_features: 文本特征 [N x D]
    
    Returns:
        results: 包含所有指标的字典
    """
    print(f"\n{'='*60}")
    print("正在计算检索指标...")
    print(f"{'='*60}")
    
    # 计算相似度矩阵
    # image_features和text_features已经归一化，所以点积即为余弦相似度
    similarity_matrix = image_features @ text_features.T
    
    results = {}
    
    # Image-to-Text检索
    print("\n[Image-to-Text Retrieval]")
    i2t_r1 = compute_recall_at_k(similarity_matrix, 1, mode="i2t")
    i2t_r5 = compute_recall_at_k(similarity_matrix, 5, mode="i2t")
    i2t_r10 = compute_recall_at_k(similarity_matrix, 10, mode="i2t")
    i2t_mean = (i2t_r1 + i2t_r5 + i2t_r10) / 3.0
    
    results["I2T_R@1"] = i2t_r1
    results["I2T_R@5"] = i2t_r5
    results["I2T_R@10"] = i2t_r10
    results["I2T_MeanRecall"] = i2t_mean
    
    print(f"  R@1:  {i2t_r1:.2f}%")
    print(f"  R@5:  {i2t_r5:.2f}%")
    print(f"  R@10: {i2t_r10:.2f}%")
    print(f"  Mean: {i2t_mean:.2f}%")
    
    # Text-to-Image检索
    print("\n[Text-to-Image Retrieval]")
    t2i_r1 = compute_recall_at_k(similarity_matrix, 1, mode="t2i")
    t2i_r5 = compute_recall_at_k(similarity_matrix, 5, mode="t2i")
    t2i_r10 = compute_recall_at_k(similarity_matrix, 10, mode="t2i")
    t2i_mean = (t2i_r1 + t2i_r5 + t2i_r10) / 3.0
    
    results["T2I_R@1"] = t2i_r1
    results["T2I_R@5"] = t2i_r5
    results["T2I_R@10"] = t2i_r10
    results["T2I_MeanRecall"] = t2i_mean
    
    print(f"  R@1:  {t2i_r1:.2f}%")
    print(f"  R@5:  {t2i_r5:.2f}%")
    print(f"  R@10: {t2i_r10:.2f}%")
    print(f"  Mean: {t2i_mean:.2f}%")
    
    # 总体指标
    print("\n[Overall Metrics]")
    overall_r1 = (i2t_r1 + t2i_r1) / 2.0
    overall_r5 = (i2t_r5 + t2i_r5) / 2.0
    overall_r10 = (i2t_r10 + t2i_r10) / 2.0
    overall_mean = (i2t_mean + t2i_mean) / 2.0
    
    results["R@1"] = overall_r1
    results["R@5"] = overall_r5
    results["R@10"] = overall_r10
    results["MeanRecall"] = overall_mean
    
    print(f"  R@1:  {overall_r1:.2f}%")
    print(f"  R@5:  {overall_r5:.2f}%")
    print(f"  R@10: {overall_r10:.2f}%")
    print(f"  MeanRecall: {overall_mean:.2f}%")
    
    return results


def save_results(results, args):
    """保存评估结果"""
    if args.output_dir is None:
        output_dir = Path(args.checkpoint).parent if args.checkpoint else Path(".")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存结果到文本文件
    if args.output_name:
        # 使用用户指定的文件名
        result_file = output_dir / f"{args.output_name}.txt"
    else:
        # 默认使用checkpoint名称
        checkpoint_name = Path(args.checkpoint).stem if args.checkpoint else "unknown"
        result_file = output_dir / f"retrieval_results_{checkpoint_name}.txt"
    
    with open(result_file, "w") as f:
        f.write("SARVLM Retrieval Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Data: {args.data_csv}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Precision: {args.precision}\n")
        f.write("\n" + "=" * 50 + "\n\n")
        
        f.write("[Image-to-Text Retrieval]\n")
        f.write(f"  R@1:  {results['I2T_R@1']:.2f}%\n")
        f.write(f"  R@5:  {results['I2T_R@5']:.2f}%\n")
        f.write(f"  R@10: {results['I2T_R@10']:.2f}%\n")
        f.write(f"  MeanRecall: {results['I2T_MeanRecall']:.2f}%\n")
        
        f.write("\n[Text-to-Image Retrieval]\n")
        f.write(f"  R@1:  {results['T2I_R@1']:.2f}%\n")
        f.write(f"  R@5:  {results['T2I_R@5']:.2f}%\n")
        f.write(f"  R@10: {results['T2I_R@10']:.2f}%\n")
        f.write(f"  MeanRecall: {results['T2I_MeanRecall']:.2f}%\n")
        
        f.write("\n[Overall]\n")
        f.write(f"  R@1:  {results['R@1']:.2f}%\n")
        f.write(f"  R@5:  {results['R@5']:.2f}%\n")
        f.write(f"  R@10: {results['R@10']:.2f}%\n")
        f.write(f"  MeanRecall: {results['MeanRecall']:.2f}%\n")
    
    print(f"\n结果已保存至: {result_file}")
    
    return result_file


def main():
    """主函数"""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("SARVLM 图文检索评估")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  模型: {args.model}")
    print(f"  权重: {args.checkpoint}")
    print(f"  数据: {args.data_csv}")
    print(f"  设备: {args.device}")
    print(f"  精度: {args.precision}")
    print(f"  批大小: {args.batch_size}")
    
    # 1. 加载模型
    model, preprocess, tokenizer = load_model(args)
    
    # 2. 加载数据
    images, captions = load_data(args)
    
    # 3. 提取特征
    image_features, text_features = extract_features(
        model, preprocess, tokenizer, images, captions, args
    )
    
    # 4. 保存特征 (可选)
    if args.save_features:
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = Path(args.checkpoint).stem if args.checkpoint else "unknown"
        
        torch.save(image_features, output_dir / f"image_features_{checkpoint_name}.pt")
        torch.save(text_features, output_dir / f"text_features_{checkpoint_name}.pt")
        print(f"特征已保存至: {output_dir}")
    
    # 5. 评估检索性能
    results = evaluate_retrieval(image_features, text_features)
    
    # 6. 保存结果
    save_results(results, args)
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

