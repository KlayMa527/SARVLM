#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SARVLM Caption BERTScore 评估脚本

评估指标：
- BERTScore Precision: 生成文本中有多少内容与参考文本语义相似
- BERTScore Recall: 参考文本中有多少内容被生成文本覆盖
- BERTScore F1: Precision和Recall的调和平均

使用方式：
    python eval_bertscore.py \
        --input-csv /path/to/generated_captions.csv \
        --model-type roberta-large \
        --batch-size 64 \
        --output-dir /path/to/output

CSV文件格式要求：
    - ground_truth: 参考caption列名
    - generated: 生成caption列名
    (可通过参数自定义列名)
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 尝试导入bert_score
try:
    from bert_score import score as bert_score_fn
    from bert_score import BERTScorer
    HAS_BERT_SCORE = True
except ImportError:
    HAS_BERT_SCORE = False
    print("警告: 未安装bert_score库，请运行: pip install bert-score")

import torch


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="SARVLM Caption BERTScore 评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 输入相关参数
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="包含生成caption的CSV文件路径"
    )
    parser.add_argument(
        "--ref-col",
        type=str,
        default="ground_truth",
        help="参考caption的列名"
    )
    parser.add_argument(
        "--hyp-col",
        type=str,
        default="generated",
        help="生成caption的列名"
    )
    parser.add_argument(
        "--image-col",
        type=str,
        default="image_path",
        help="图像路径的列名 (可选，用于保存详细结果)"
    )
    
    # BERTScore 相关参数
    parser.add_argument(
        "--model-type",
        type=str,
        default="roberta-large",
        help="BERTScore使用的预训练模型 (如: roberta-large, bert-base-uncased, microsoft/deberta-xlarge-mnli)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="使用预训练模型的哪一层 (默认使用最佳层)"
    )
    parser.add_argument(
        "--rescale-with-baseline",
        action="store_true",
        help="是否使用baseline进行重新缩放 (推荐用于跨模型比较)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="文本语言 (en, zh, multi等)"
    )
    parser.add_argument(
        "--idf",
        action="store_true",
        help="是否使用IDF加权"
    )
    
    # 运行相关参数
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批处理大小"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="是否显示详细进度"
    )
    
    # 输出相关参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="结果保存目录 (默认为输入CSV同级目录)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="输出文件名前缀 (不含扩展名)"
    )
    parser.add_argument(
        "--save-per-sample",
        action="store_true",
        help="是否保存每个样本的详细分数"
    )
    
    return parser.parse_args()


def load_captions(csv_path: str, ref_col: str, hyp_col: str, image_col: str = None) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """
    从CSV文件加载参考和生成的caption
    
    Args:
        csv_path: CSV文件路径
        ref_col: 参考caption列名
        hyp_col: 生成caption列名
        image_col: 图像路径列名 (可选)
    
    Returns:
        references: 参考caption列表
        hypotheses: 生成caption列表
        image_paths: 图像路径列表 (如果提供)
    """
    references = []
    hypotheses = []
    image_paths = []
    
    print(f"\n{'='*60}")
    print(f"正在加载数据: {csv_path}")
    print(f"{'='*60}")
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        # 检查列名是否存在
        fieldnames = reader.fieldnames
        if ref_col not in fieldnames:
            raise ValueError(f"参考列 '{ref_col}' 不在CSV文件中。可用列: {fieldnames}")
        if hyp_col not in fieldnames:
            raise ValueError(f"生成列 '{hyp_col}' 不在CSV文件中。可用列: {fieldnames}")
        
        has_image_col = image_col and image_col in fieldnames
        
        for row in reader:
            ref = row[ref_col].strip()
            hyp = row[hyp_col].strip()
            
            # 跳过空行
            if not ref or not hyp:
                continue
            
            references.append(ref)
            hypotheses.append(hyp)
            
            if has_image_col:
                image_paths.append(row[image_col])
    
    print(f"成功加载 {len(references)} 条样本")
    print(f"参考caption示例: {references[0][:80]}...")
    print(f"生成caption示例: {hypotheses[0][:80]}...")
    
    return references, hypotheses, image_paths if has_image_col else None


def compute_bertscore(
    references: List[str],
    hypotheses: List[str],
    model_type: str = "roberta-large",
    num_layers: int = None,
    batch_size: int = 64,
    device: str = "cuda",
    lang: str = "en",
    rescale_with_baseline: bool = False,
    idf: bool = False,
    verbose: bool = False,
) -> Dict[str, any]:
    """
    计算BERTScore
    
    Args:
        references: 参考caption列表
        hypotheses: 生成caption列表
        model_type: 预训练模型名称
        num_layers: 使用的层数
        batch_size: 批处理大小
        device: 运行设备
        lang: 文本语言
        rescale_with_baseline: 是否使用baseline重新缩放
        idf: 是否使用IDF加权
        verbose: 是否显示详细信息
    
    Returns:
        包含BERTScore结果的字典
    """
    if not HAS_BERT_SCORE:
        raise ImportError("请先安装bert_score: pip install bert-score")
    
    print(f"\n{'='*60}")
    print(f"正在计算 BERTScore")
    print(f"{'='*60}")
    print(f"模型: {model_type}")
    print(f"设备: {device}")
    print(f"批大小: {batch_size}")
    print(f"语言: {lang}")
    print(f"使用Baseline缩放: {rescale_with_baseline}")
    print(f"使用IDF加权: {idf}")
    
    # 计算BERTScore
    P, R, F1 = bert_score_fn(
        cands=hypotheses,
        refs=references,
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        device=device,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        idf=idf,
        verbose=verbose,
    )
    
    # 转换为numpy数组
    P = P.numpy()
    R = R.numpy()
    F1 = F1.numpy()
    
    # 计算统计量
    results = {
        "precision": {
            "mean": float(np.mean(P)),
            "std": float(np.std(P)),
            "min": float(np.min(P)),
            "max": float(np.max(P)),
            "median": float(np.median(P)),
        },
        "recall": {
            "mean": float(np.mean(R)),
            "std": float(np.std(R)),
            "min": float(np.min(R)),
            "max": float(np.max(R)),
            "median": float(np.median(R)),
        },
        "f1": {
            "mean": float(np.mean(F1)),
            "std": float(np.std(F1)),
            "min": float(np.min(F1)),
            "max": float(np.max(F1)),
            "median": float(np.median(F1)),
        },
        "per_sample": {
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F1.tolist(),
        },
        "num_samples": len(references),
    }
    
    return results


def print_results(results: Dict, model_type: str):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"BERTScore 评估结果 (模型: {model_type})")
    print(f"{'='*60}")
    print(f"样本数: {results['num_samples']}")
    print()
    
    print(f"[Precision (生成文本与参考文本的语义覆盖度)]")
    print(f"  Mean:   {results['precision']['mean']:.4f}")
    print(f"  Std:    {results['precision']['std']:.4f}")
    print(f"  Median: {results['precision']['median']:.4f}")
    print(f"  Range:  [{results['precision']['min']:.4f}, {results['precision']['max']:.4f}]")
    print()
    
    print(f"[Recall (参考文本被生成文本覆盖的程度)]")
    print(f"  Mean:   {results['recall']['mean']:.4f}")
    print(f"  Std:    {results['recall']['std']:.4f}")
    print(f"  Median: {results['recall']['median']:.4f}")
    print(f"  Range:  [{results['recall']['min']:.4f}, {results['recall']['max']:.4f}]")
    print()
    
    print(f"[F1 Score (Precision和Recall的调和平均)]")
    print(f"  Mean:   {results['f1']['mean']:.4f}")
    print(f"  Std:    {results['f1']['std']:.4f}")
    print(f"  Median: {results['f1']['median']:.4f}")
    print(f"  Range:  [{results['f1']['min']:.4f}, {results['f1']['max']:.4f}]")


def save_results(
    results: Dict,
    args,
    references: List[str] = None,
    hypotheses: List[str] = None,
    image_paths: List[str] = None,
):
    """保存评估结果"""
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.input_csv).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定输出文件名
    if args.output_name:
        base_name = args.output_name
    else:
        input_name = Path(args.input_csv).stem
        base_name = f"bertscore_{input_name}"
    
    # 保存摘要结果 (txt)
    txt_file = output_dir / f"{base_name}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("SARVLM Caption BERTScore Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input CSV: {args.input_csv}\n")
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Language: {args.lang}\n")
        f.write(f"Rescale with Baseline: {args.rescale_with_baseline}\n")
        f.write(f"Use IDF: {args.idf}\n")
        f.write(f"Number of Samples: {results['num_samples']}\n")
        f.write("\n" + "=" * 50 + "\n\n")
        
        f.write("[Precision]\n")
        f.write(f"  Mean:   {results['precision']['mean']:.4f}\n")
        f.write(f"  Std:    {results['precision']['std']:.4f}\n")
        f.write(f"  Median: {results['precision']['median']:.4f}\n")
        f.write(f"  Range:  [{results['precision']['min']:.4f}, {results['precision']['max']:.4f}]\n\n")
        
        f.write("[Recall]\n")
        f.write(f"  Mean:   {results['recall']['mean']:.4f}\n")
        f.write(f"  Std:    {results['recall']['std']:.4f}\n")
        f.write(f"  Median: {results['recall']['median']:.4f}\n")
        f.write(f"  Range:  [{results['recall']['min']:.4f}, {results['recall']['max']:.4f}]\n\n")
        
        f.write("[F1 Score]\n")
        f.write(f"  Mean:   {results['f1']['mean']:.4f}\n")
        f.write(f"  Std:    {results['f1']['std']:.4f}\n")
        f.write(f"  Median: {results['f1']['median']:.4f}\n")
        f.write(f"  Range:  [{results['f1']['min']:.4f}, {results['f1']['max']:.4f}]\n")
    
    print(f"\n结果已保存至: {txt_file}")
    
    # 保存JSON结果 (不含per_sample)
    json_file = output_dir / f"{base_name}.json"
    json_results = {
        "config": {
            "input_csv": str(args.input_csv),
            "model_type": args.model_type,
            "lang": args.lang,
            "rescale_with_baseline": args.rescale_with_baseline,
            "idf": args.idf,
            "device": args.device,
            "batch_size": args.batch_size,
        },
        "metrics": {
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
        },
        "num_samples": results["num_samples"],
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON结果已保存至: {json_file}")
    
    # 保存每个样本的详细分数 (可选)
    if args.save_per_sample and references and hypotheses:
        detail_file = output_dir / f"{base_name}_per_sample.csv"
        with open(detail_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            
            if image_paths:
                writer.writerow(["image_path", "reference", "hypothesis", "precision", "recall", "f1"])
            else:
                writer.writerow(["reference", "hypothesis", "precision", "recall", "f1"])
            
            per_sample = results["per_sample"]
            for i in range(len(references)):
                row = []
                if image_paths:
                    row.append(image_paths[i])
                row.extend([
                    references[i],
                    hypotheses[i],
                    f"{per_sample['precision'][i]:.4f}",
                    f"{per_sample['recall'][i]:.4f}",
                    f"{per_sample['f1'][i]:.4f}",
                ])
                writer.writerow(row)
        
        print(f"每样本详细结果已保存至: {detail_file}")
    
    return txt_file


def main():
    """主函数"""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("SARVLM Caption BERTScore 评估")
    print("=" * 60)
    
    # 检查bert_score是否安装
    if not HAS_BERT_SCORE:
        print("\n错误: 请先安装bert_score库")
        print("运行: pip install bert-score")
        sys.exit(1)
    
    # 检查输入文件
    if not os.path.exists(args.input_csv):
        print(f"\n错误: 输入文件不存在: {args.input_csv}")
        sys.exit(1)
    
    print(f"\n配置:")
    print(f"  输入文件: {args.input_csv}")
    print(f"  参考列: {args.ref_col}")
    print(f"  生成列: {args.hyp_col}")
    print(f"  模型: {args.model_type}")
    print(f"  设备: {args.device}")
    print(f"  批大小: {args.batch_size}")
    
    # 1. 加载数据
    references, hypotheses, image_paths = load_captions(
        args.input_csv,
        args.ref_col,
        args.hyp_col,
        args.image_col,
    )
    
    if len(references) == 0:
        print("\n错误: 未找到有效样本")
        sys.exit(1)
    
    # 2. 计算BERTScore
    results = compute_bertscore(
        references=references,
        hypotheses=hypotheses,
        model_type=args.model_type,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        device=args.device,
        lang=args.lang,
        rescale_with_baseline=args.rescale_with_baseline,
        idf=args.idf,
        verbose=args.verbose,
    )
    
    # 3. 打印结果
    print_results(results, args.model_type)
    
    # 4. 保存结果
    save_results(
        results,
        args,
        references=references if args.save_per_sample else None,
        hypotheses=hypotheses if args.save_per_sample else None,
        image_paths=image_paths if args.save_per_sample else None,
    )
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()



