#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SARVLM 零样本分类评估脚本

评估指标：
- Top-1 Accuracy: 预测的最高概率类别是否正确
- Top-3 Accuracy: 正确类别是否在前3个预测中
- Top-5 Accuracy: 正确类别是否在前5个预测中
- Per-class Accuracy: 每个类别的准确率
- Mean Per-class Accuracy: 所有类别准确率的平均值
- Confusion Matrix: 混淆矩阵

数据集格式：
    data_root/
    ├── class1/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── class2/
    │   ├── image3.jpg
    │   └── image4.jpg
    └── ...

Usage:
    python eval_zeroshot.py \
        --model ViT-L-14 \
        --checkpoint /path/to/checkpoint.pt \
        --data-root /path/to/dataset \
        --batch-size 32 \
        --device cuda
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 尝试导入GeoTiff支持库
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

# 如果都没有安装，在遇到tif文件时会提示
if not HAS_RASTERIO and not HAS_TIFFFILE:
    import warnings
    warnings.warn(
        "未安装rasterio或tifffile库，可能无法正确加载GeoTiff文件。\n"
        "建议安装: pip install rasterio 或 pip install tifffile"
    )

# 添加src目录到路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import open_clip
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.zero_shot_classifier import build_zero_shot_classifier


def load_image(img_path: str) -> Image.Image:
    """
    加载图像，支持多种格式包括GeoTiff
    
    Args:
        img_path: 图像文件路径
    
    Returns:
        PIL Image对象 (RGB模式)
    """
    img_path = str(img_path)
    suffix = Path(img_path).suffix.lower()
    
    # 对于tif/tiff文件，尝试使用专门的库
    if suffix in ['.tif', '.tiff']:
        # 尝试使用rasterio (最佳GeoTiff支持)
        if HAS_RASTERIO:
            try:
                with rasterio.open(img_path) as src:
                    # 读取所有波段
                    data = src.read()
                    
                    # 处理不同波段数的情况
                    if data.shape[0] == 1:
                        # 单波段，复制为3通道
                        data = np.repeat(data, 3, axis=0)
                    elif data.shape[0] == 2:
                        # 双波段(如VV, VH)，添加第三个通道
                        third_channel = (data[0] + data[1]) / 2
                        data = np.stack([data[0], data[1], third_channel], axis=0)
                    elif data.shape[0] > 3:
                        # 多于3个波段，只取前3个
                        data = data[:3]
                    
                    # 转换为HWC格式
                    data = np.transpose(data, (1, 2, 0))
                    
                    # 归一化到0-255
                    if data.dtype != np.uint8:
                        # 处理float或其他类型
                        data_min, data_max = np.nanmin(data), np.nanmax(data)
                        if data_max > data_min:
                            data = (data - data_min) / (data_max - data_min) * 255
                        else:
                            data = np.zeros_like(data)
                        data = np.clip(data, 0, 255).astype(np.uint8)
                    
                    # 处理NaN值
                    data = np.nan_to_num(data, nan=0)
                    
                    return Image.fromarray(data.astype(np.uint8), mode='RGB')
            except Exception as e:
                pass  # 尝试其他方法
        
        # 尝试使用tifffile
        if HAS_TIFFFILE:
            try:
                data = tifffile.imread(img_path)
                
                # 处理不同维度
                if data.ndim == 2:
                    # 单通道
                    data = np.stack([data, data, data], axis=-1)
                elif data.ndim == 3:
                    if data.shape[0] in [1, 2, 3, 4] and data.shape[0] < data.shape[1]:
                        # CHW格式
                        data = np.transpose(data, (1, 2, 0))
                    
                    if data.shape[-1] == 1:
                        data = np.repeat(data, 3, axis=-1)
                    elif data.shape[-1] == 2:
                        third_channel = (data[..., 0] + data[..., 1]) / 2
                        data = np.stack([data[..., 0], data[..., 1], third_channel], axis=-1)
                    elif data.shape[-1] > 3:
                        data = data[..., :3]
                
                # 归一化到0-255
                if data.dtype != np.uint8:
                    data_min, data_max = np.nanmin(data), np.nanmax(data)
                    if data_max > data_min:
                        data = (data - data_min) / (data_max - data_min) * 255
                    else:
                        data = np.zeros_like(data)
                    data = np.clip(data, 0, 255).astype(np.uint8)
                
                data = np.nan_to_num(data, nan=0)
                return Image.fromarray(data.astype(np.uint8), mode='RGB')
            except Exception as e:
                pass  # 尝试PIL
    
    # 默认使用PIL
    img = Image.open(img_path)
    
    # 确保是RGB模式
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img


# ============================================================
# SAR图像零样本分类的文本模板
# ============================================================
SAR_TEMPLATES = [
    "a SAR image of {}.",
    "a radar image of {}.",
    "a synthetic aperture radar image of {}.",
    "SAR imagery showing {}.",
    "a satellite SAR image of {}.",
    "radar remote sensing image of {}.",
    "{}.",
    "a photo of {}.",
    "an image of {}.",
]

OPTICAL_TEMPLATES = [
    "a satellite image of {}.",
    "an aerial image of {}.",
    "an aerial photo of {}.",
    "a remote sensing image of {}.",
    "an overhead view of {}.",
    "a high-resolution satellite image of {}.",
    "imagery showing {}.",
    "an image of {}.",
    "{}.",
]

SAR_OPTICAL_TEMPLATE = [
    "a SAR image of {}.",
    "a radar image of {}.",
    "a synthetic aperture radar image of {}.",
    "SAR imagery showing {}.",
    "a satellite SAR image of {}.",
    "radar remote sensing image of {}.",
    "a satellite image of {}.",
    "an aerial image of {}.",
    "an aerial photo of {}.",
    "a remote sensing image of {}.",
    "an overhead view of {}.",
    "a high-resolution satellite image of {}.",
    "a photo of {}.",
    "an image of {}.",
    "{}.",
]



# 可以根据具体数据集定制的模板
ISPRS_TEMPLATES = [
    "a SAR image of {}.",
    "a radar image showing {}.",
    "synthetic aperture radar image of {}.",
    "SAR land cover image of {}.",
    "{}.",
]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="SARVLM 零样本分类评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型相关参数
    parser.add_argument(
        "--model", 
        type=str, 
        default="ViT-L-14",
        help="模型架构名称 (如: ViT-L-14, ViT-B-32, coca_ViT-L-14)"
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
        "--data-root", 
        type=str, 
        required=True,
        help="数据集根目录 (每个子目录为一个类别)"
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="类别名称列表 (默认使用目录名)"
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="sar",
        choices=["sar", "isprs", "simple", 'optical', 'sar_opt'],
        help="文本模板类型: sar (通用SAR), isprs (ISPRS数据集), simple (简单)"
    )
    parser.add_argument(
        "--custom-templates",
        type=str,
        default=None,
        help="自定义模板JSON文件路径"
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
        help="输出结果文件名 (不含扩展名)"
    )
    parser.add_argument(
        "--save-predictions", 
        action="store_true",
        help="是否保存每个样本的预测结果"
    )
    
    return parser.parse_args()


def get_templates(template_type: str, custom_path: str = None) -> List[str]:
    """获取文本模板"""
    if custom_path and os.path.exists(custom_path):
        with open(custom_path, "r") as f:
            return json.load(f)
    
    if template_type == "sar":
        return SAR_TEMPLATES
    elif template_type == "isprs":
        return ISPRS_TEMPLATES
    elif template_type =="optical":
        return OPTICAL_TEMPLATES
    elif template_type == "sar_opt":
        return SAR_OPTICAL_TEMPLATE
    else:  # simple
        return ["a photo of {}."]


def load_model(args):
    """加载模型和预处理器"""
    print(f"\n{'='*60}")
    print(f"正在加载模型: {args.model}")
    print(f"{'='*60}")
    
    # 检查是否是OpenAI原生CLIP权重（真正的TorchScript格式）
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
        elif args.checkpoint:
            print(f"警告: checkpoint文件不存在: {args.checkpoint}")
    
    model.eval()
    tokenizer = get_tokenizer(args.model)
    
    return model, preprocess, tokenizer


def load_dataset(data_root: str, class_names: List[str] = None) -> Tuple[List[str], List[int], List[str]]:
    """
    加载数据集
    
    Args:
        data_root: 数据集根目录
        class_names: 指定的类别名称列表
    
    Returns:
        image_paths: 图像路径列表
        labels: 标签列表
        classes: 类别名称列表
    """
    print(f"\n{'='*60}")
    print(f"正在加载数据集: {data_root}")
    print(f"{'='*60}")
    
    data_root = Path(data_root)
    
    # 获取所有类别目录
    if class_names is None:
        class_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
        classes = [d.name for d in class_dirs]
    else:
        classes = class_names
        class_dirs = [data_root / c for c in classes]
    
    print(f"发现 {len(classes)} 个类别:")
    for i, c in enumerate(classes):
        print(f"  [{i}] {c}")
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif'}
    
    image_paths = []
    labels = []
    
    for class_idx, class_dir in enumerate(class_dirs):
        if not class_dir.exists():
            print(f"警告: 类别目录不存在: {class_dir}")
            continue
            
        class_images = []
        for ext in image_extensions:
            class_images.extend(class_dir.glob(f"*{ext}"))
            class_images.extend(class_dir.glob(f"*{ext.upper()}"))
        
        for img_path in class_images:
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    print(f"\n总样本数: {len(image_paths)}")
    
    # 统计每个类别的样本数
    class_counts = defaultdict(int)
    for label in labels:
        class_counts[label] += 1
    
    print("每个类别的样本数:")
    for class_idx, count in sorted(class_counts.items()):
        print(f"  {classes[class_idx]}: {count}")
    
    return image_paths, labels, classes


@torch.no_grad()
def build_classifier(model, tokenizer, classes: List[str], templates: List[str], device: str):
    """构建零样本分类器权重"""
    print(f"\n{'='*60}")
    print("正在构建零样本分类器...")
    print(f"{'='*60}")
    print(f"类别数: {len(classes)}")
    print(f"模板数: {len(templates)}")
    
    classifier = build_zero_shot_classifier(
        model=model,
        tokenizer=tokenizer,
        classnames=classes,
        templates=templates,
        device=device,
        use_tqdm=True,
    )
    
    print(f"分类器权重维度: {classifier.shape}")
    return classifier


@torch.no_grad()
def evaluate(model, preprocess, classifier, image_paths, labels, classes, args):
    """评估零样本分类性能"""
    print(f"\n{'='*60}")
    print("正在评估...")
    print(f"{'='*60}")
    
    device = args.device
    batch_size = args.batch_size
    num_classes = len(classes)
    
    all_predictions = []
    all_probs = []
    all_labels = labels
    
    # 批量处理
    for i in tqdm(range(0, len(image_paths), batch_size), desc="评估进度"):
        batch_paths = image_paths[i:i + batch_size]
        
        # 加载和预处理图像
        batch_tensors = []
        for img_path in batch_paths:
            try:
                img = load_image(img_path)
                img_tensor = preprocess(img)
                batch_tensors.append(img_tensor)
            except Exception as e:
                print(f"警告: 无法加载图像 {img_path}: {e}")
                img_tensor = torch.zeros(3, 224, 224)
                batch_tensors.append(img_tensor)
        
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # 提取图像特征
        image_features = model.encode_image(batch_tensor, normalize=True)
        
        # 计算logits
        logits = 100.0 * image_features @ classifier
        probs = logits.softmax(dim=-1)
        
        predictions = logits.argmax(dim=-1).cpu().numpy()
        all_predictions.extend(predictions.tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算指标
    results = compute_metrics(all_predictions, all_labels, all_probs, classes)
    
    return results, all_predictions, all_probs


def compute_metrics(predictions, labels, probs, classes):
    """计算评估指标"""
    num_classes = len(classes)
    num_samples = len(labels)
    
    results = {}
    
    # Top-1 Accuracy
    top1_correct = (predictions == labels).sum()
    top1_acc = top1_correct / num_samples * 100.0
    results["top1_accuracy"] = top1_acc
    
    # Top-3 Accuracy (如果类别数>=3)
    if num_classes >= 3:
        top3_indices = np.argsort(probs, axis=1)[:, -3:]
        top3_correct = sum(labels[i] in top3_indices[i] for i in range(num_samples))
        top3_acc = top3_correct / num_samples * 100.0
        results["top3_accuracy"] = top3_acc
    else:
        results["top3_accuracy"] = None
    
    # Top-5 Accuracy (如果类别数>=5)
    if num_classes >= 5:
        top5_indices = np.argsort(probs, axis=1)[:, -5:]
        top5_correct = sum(labels[i] in top5_indices[i] for i in range(num_samples))
        top5_acc = top5_correct / num_samples * 100.0
        results["top5_accuracy"] = top5_acc
    else:
        results["top5_accuracy"] = None
    
    # Per-class Accuracy
    per_class_acc = {}
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    
    for pred, label in zip(predictions, labels):
        per_class_total[label] += 1
        if pred == label:
            per_class_correct[label] += 1
    
    for class_idx in range(num_classes):
        if per_class_total[class_idx] > 0:
            acc = per_class_correct[class_idx] / per_class_total[class_idx] * 100.0
        else:
            acc = 0.0
        per_class_acc[classes[class_idx]] = acc
    
    results["per_class_accuracy"] = per_class_acc
    
    # Mean per-class Accuracy (只计算有样本的类别)
    valid_accs = [acc for class_idx, acc in enumerate(per_class_acc.values()) if per_class_total[class_idx] > 0]
    mean_per_class_acc = np.mean(valid_accs) if valid_accs else 0.0
    results["mean_per_class_accuracy"] = mean_per_class_acc
    
    # Confusion Matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(predictions, labels):
        confusion_matrix[label, pred] += 1
    results["confusion_matrix"] = confusion_matrix.tolist()
    
    return results


def print_results(results, classes):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    
    print(f"\n[Overall Metrics]")
    print(f"  Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    if results["top3_accuracy"] is not None:
        print(f"  Top-3 Accuracy: {results['top3_accuracy']:.2f}%")
    if results["top5_accuracy"] is not None:
        print(f"  Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    print(f"  Mean Per-class Accuracy: {results['mean_per_class_accuracy']:.2f}%")
    
    print(f"\n[Per-class Accuracy]")
    for class_name, acc in results["per_class_accuracy"].items():
        print(f"  {class_name}: {acc:.2f}%")
    
    # 打印混淆矩阵
    print(f"\n[Confusion Matrix]")
    cm = np.array(results["confusion_matrix"])
    
    # 表头
    header = "True\\Pred".ljust(25) + "".join([c[:8].ljust(10) for c in classes])
    print(header)
    print("-" * len(header))
    
    for i, class_name in enumerate(classes):
        row = class_name[:23].ljust(25) + "".join([str(cm[i, j]).ljust(10) for j in range(len(classes))])
        print(row)


def save_results(results, classes, args, predictions=None, probs=None, image_paths=None):
    """保存评估结果"""
    if args.output_dir is None:
        output_dir = Path(args.checkpoint).parent if args.checkpoint else Path(".")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定输出文件名
    if args.output_name:
        base_name = args.output_name
    else:
        checkpoint_name = Path(args.checkpoint).stem if args.checkpoint else "unknown"
        dataset_name = Path(args.data_root).name
        base_name = f"zeroshot_{checkpoint_name}_{dataset_name}"
    
    # 保存文本结果
    result_file = output_dir / f"{base_name}.txt"
    with open(result_file, "w") as f:
        f.write("SARVLM Zero-shot Classification Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.data_root}\n")
        f.write(f"Templates: {args.templates}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Precision: {args.precision}\n")
        f.write("\n" + "=" * 50 + "\n\n")
        
        f.write("[Overall Metrics]\n")
        f.write(f"  Top-1 Accuracy: {results['top1_accuracy']:.2f}%\n")
        if results["top3_accuracy"] is not None:
            f.write(f"  Top-3 Accuracy: {results['top3_accuracy']:.2f}%\n")
        if results["top5_accuracy"] is not None:
            f.write(f"  Top-5 Accuracy: {results['top5_accuracy']:.2f}%\n")
        f.write(f"  Mean Per-class Accuracy: {results['mean_per_class_accuracy']:.2f}%\n")
        
        f.write("\n[Per-class Accuracy]\n")
        for class_name, acc in results["per_class_accuracy"].items():
            f.write(f"  {class_name}: {acc:.2f}%\n")
        
        f.write("\n[Confusion Matrix]\n")
        cm = np.array(results["confusion_matrix"])
        header = "True\\Pred".ljust(25) + "".join([c[:8].ljust(10) for c in classes])
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for i, class_name in enumerate(classes):
            row = class_name[:23].ljust(25) + "".join([str(cm[i, j]).ljust(10) for j in range(len(classes))])
            f.write(row + "\n")
    
    print(f"\n结果已保存至: {result_file}")
    
    # 保存JSON结果
    json_file = output_dir / f"{base_name}.json"
    json_results = {
        "config": {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "dataset": args.data_root,
            "templates": args.templates,
        },
        "metrics": {
            "top1_accuracy": results["top1_accuracy"],
            "top3_accuracy": results["top3_accuracy"],
            "top5_accuracy": results["top5_accuracy"],
            "mean_per_class_accuracy": results["mean_per_class_accuracy"],
            "per_class_accuracy": results["per_class_accuracy"],
        },
        "confusion_matrix": results["confusion_matrix"],
        "classes": classes,
    }
    with open(json_file, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON结果已保存至: {json_file}")
    
    # 保存预测结果 (可选)
    if args.save_predictions and predictions is not None:
        pred_file = output_dir / f"{base_name}_predictions.json"
        pred_results = []
        for i, (img_path, pred, prob) in enumerate(zip(image_paths, predictions, probs)):
            pred_results.append({
                "image": img_path,
                "prediction": classes[pred],
                "prediction_idx": int(pred),
                "probabilities": {classes[j]: float(prob[j]) for j in range(len(classes))},
            })
        with open(pred_file, "w") as f:
            json.dump(pred_results, f, indent=2)
        print(f"预测结果已保存至: {pred_file}")
    
    return result_file


def main():
    """主函数"""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("SARVLM 零样本分类评估")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  模型: {args.model}")
    print(f"  权重: {args.checkpoint}")
    print(f"  数据: {args.data_root}")
    print(f"  模板: {args.templates}")
    print(f"  设备: {args.device}")
    print(f"  精度: {args.precision}")
    print(f"  批大小: {args.batch_size}")
    
    # 1. 加载模型
    model, preprocess, tokenizer = load_model(args)
    
    # 2. 加载数据集
    image_paths, labels, classes = load_dataset(args.data_root, args.class_names)
    
    # 3. 获取模板
    templates = get_templates(args.templates, args.custom_templates)
    print(f"\n使用的模板:")
    for t in templates:
        print(f"  - {t}")
    
    # 4. 构建分类器
    classifier = build_classifier(model, tokenizer, classes, templates, args.device)
    
    # 5. 评估
    results, predictions, probs = evaluate(
        model, preprocess, classifier, image_paths, labels, classes, args
    )
    
    # 6. 打印结果
    print_results(results, classes)
    
    # 7. 保存结果
    save_results(results, classes, args, predictions, probs, image_paths)
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

