#!/usr/bin/env python3
"""
CoCa模型推理和评估脚本 - 简化版
对测试图像进行caption生成，并计算评估指标
"""

import os
import sys
import argparse
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import json

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import open_clip
from open_clip import create_model_and_transforms, load_checkpoint, get_tokenizer

# 评估指标库
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


def load_model(model_name, checkpoint_path, device='cuda'):
    """加载模型和checkpoint权重"""
    print(f"Loading model: {model_name}")
    model, _, preprocess = create_model_and_transforms(
        model_name,
        pretrained=None,
        precision='fp32',
        device=device,
        jit=False,
    )
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, strict=False, weights_only=False)
    model = model.to(device)
    model.eval()
    
    return model, preprocess


def generate_captions(model, tokenizer, image_paths, preprocess, device='cuda', batch_size=8):
    """对图像生成captions"""
    generated_captions = []
    
    print("Generating captions...")
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            # 加载和预处理图像
            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                image_tensor = preprocess(image).unsqueeze(0)
                batch_images.append(image_tensor)
            
            batch_images = torch.cat(batch_images, dim=0).to(device)
            
            # 生成captions
            generated_tokens = model.generate(
                batch_images,
                seq_len=77,
                max_seq_len=77,
                temperature=1.0,
                generation_type="beam_search",
                num_beams=6,
                num_beam_groups=3,
                min_seq_len=5,
            )
            
            # 解码tokens到文本
            for tokens in generated_tokens:
                if isinstance(tokens, torch.Tensor):
                    tokens_np = tokens.cpu().numpy()
                else:
                    tokens_np = tokens
                
                # 移除padding和特殊tokens
                valid_tokens = []
                for token in tokens_np:
                    if token > 0:
                        if token == 49407:  # EOS
                            break
                        if token != 49406:  # SOS
                            valid_tokens.append(int(token))
                
                # 解码
                if hasattr(tokenizer, 'decode'):
                    caption = tokenizer.decode(valid_tokens)
                else:
                    from open_clip.tokenizer import decode
                    caption = decode(torch.tensor(valid_tokens))
                
                # 清理caption
                caption = caption.replace('<start_of_text>', '').replace('<end_of_text>', '').strip()
                caption = ' '.join(caption.split())
                generated_captions.append(caption)
    
    return generated_captions


def evaluate_captions(gts, res):
    """评估生成的captions"""
    # 准备数据格式
    gts_dict = {img_id: [caps] if not isinstance(caps, list) else caps for img_id, caps in gts.items()}
    res_dict = {img_id: [caps] if not isinstance(caps, list) else caps for img_id, caps in res.items()}
    
    # 初始化评估器
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]
    
    # 计算指标
    results = {}
    
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts_dict, res_dict)
        
        if isinstance(method, list):
            if isinstance(score, list):
                for sc, m in zip(score, method):
                    results[m] = float(sc)
            else:
                results[method[0]] = float(score)
        else:
            results[method] = float(score)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate CoCa model on test dataset')
    parser.add_argument('--model', type=str, default='coca_ViT-L-14', help='Model name')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/ubuntu/sarclip/src/logs/sarcoca_1/checkpoints/epoch_1.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--csv', type=str,
                       default='/home/ubuntu/sarclip/dataset/SARCLIP/data_csv/sarclip_val_uni_image_and_caption_a100.csv',
                       help='Path to CSV file with image paths and captions')
    parser.add_argument('--output', type=str,
                       default='/home/ubuntu/sarclip/code/results',
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载数据
    print(f"Loading data from: {args.csv}")
    df = pd.read_csv(args.csv)
    image_paths = df['imgpath'].tolist()
    gt_captions = df['caption'].tolist()
    print(f"Total samples: {len(image_paths)}")
    
    # 加载模型
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    model, preprocess = load_model(args.model, args.checkpoint, device)
    tokenizer = get_tokenizer(args.model)
    
    # 生成captions
    generated_captions = generate_captions(
        model, tokenizer, image_paths, preprocess, 
        device=device, batch_size=args.batch_size
    )
    
    # 准备评估数据
    gts = {f"image_{idx}": [gt] for idx, gt in enumerate(gt_captions)}
    res = {f"image_{idx}": [gen] for idx, gen in enumerate(generated_captions)}
    
    # 评估
    print("\nComputing evaluation metrics...")
    results = evaluate_captions(gts, res)
    
    # 打印结果
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    print("="*50)
    
    # 保存结果
    output_results = {
        'metrics': results,
        'details': [
            {
                'image_id': f"image_{idx}",
                'image_path': img_path,
                'ground_truth': gt,
                'generated': gen
            }
            for idx, (img_path, gt, gen) in enumerate(zip(image_paths, gt_captions, generated_captions))
        ]
    }
    
    # 保存JSON
    results_json_path = os.path.join(args.output, 'evaluation_results.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_json_path}")
    
    # 保存CSV
    results_df = pd.DataFrame({
        'image_path': image_paths,
        'ground_truth': gt_captions,
        'generated': generated_captions
    })
    results_csv_path = os.path.join(args.output, 'generated_captions.csv')
    results_df.to_csv(results_csv_path, index=False, na_rep='')
    print(f"Generated captions saved to: {results_csv_path}")
    
    # 保存指标摘要
    metrics_summary_path = os.path.join(args.output, 'metrics_summary.txt')
    with open(metrics_summary_path, 'w', encoding='utf-8') as f:
        f.write("Evaluation Metrics Summary\n")
        f.write("="*50 + "\n")
        for metric, score in results.items():
            f.write(f"{metric}: {score:.4f}\n")
        f.write("="*50 + "\n")
    print(f"Metrics summary saved to: {metrics_summary_path}")
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()

