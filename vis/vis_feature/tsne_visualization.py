#!/usr/bin/env python3
"""
SAR-VLM t-SNE Visualization Tool
用于可视化CLIP模型在SAR数据集上的特征分布

Author: Claude
Date: 2026-03-13
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# sklearn for t-SNE
from sklearn.manifold import TSNE

# matplotlib for plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Add src to path for open_clip
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import open_clip


# Dataset configurations
DATASET_CONFIGS = {
    'mstar': {
        'name': 'MSTAR-SOC',
        'path': '/home/maqw/SARVLM/SARVLM/data/SARATR-X/MSTAR_SOC/TEST',
        'extensions': ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff'],
    },
    'fusar': {
        'name': 'New_FUSAR',
        'path': '/home/maqw/SARVLM/SARVLM/data/SARATR-X/New_FUSAR/Val',
        'extensions': ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff'],
    },
    'sar_vsa': {
        'name': 'SAR_VSA',
        'path': '/home/maqw/SARVLM/SARVLM/data/SARATR-X/SAR_VSA/TEST',
        'extensions': ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff'],
    }
}

# IEEE论文风格的颜色方案
IEEE_COLORS = [
    '#0066CC',  # 蓝色
    '#CC0000',  # 红色
    '#009900',  # 绿色
    '#FF6600',  # 橙色
    '#9900CC',  # 紫色
    '#00CCCC',  # 青色
    '#CC6600',  # 棕色
    '#FF0066',  # 粉色
    '#666666',  # 灰色
    '#99CC00',  # 黄绿
    '#0066FF',  # 亮蓝
    '#CC3300',  # 深红
    '#00CC66',  # 翠绿
    '#FF9900',  # 金黄
    '#6600CC',  # 深紫
    '#00CCFF',  # 天蓝
    '#CC9966',  # 浅棕
    '#FF3399',  # 玫瑰
    '#336666',  # 深青
    '#99FF00',  # 荧光绿
    '#0033CC',  # 深蓝
]


def get_images_from_folder(folder_path, extensions):
    """获取文件夹中所有图像文件，按类别组织"""
    images_by_class = {}

    for class_name in sorted(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue

        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(class_path, ext)))
            images.extend(glob.glob(os.path.join(class_path, ext.upper())))

        if images:
            images_by_class[class_name] = sorted(images)

    return images_by_class


def extract_features(model, preprocess, device, images_by_class, max_samples_per_class=None):
    """
    提取图像特征

    Args:
        model: CLIP模型
        preprocess: 预处理函数
        device: 计算设备
        images_by_class: 按类别组织的图像路径字典
        max_samples_per_class: 每类最大采样数（用于大数据集）

    Returns:
        features: 特征数组 (N, D)
        labels: 标签数组 (N,)
        class_names: 类别名称列表
    """
    model.eval()

    all_features = []
    all_labels = []
    class_names = sorted(images_by_class.keys())

    print(f"Extracting features for {len(class_names)} classes...")

    with torch.no_grad():
        for class_idx, class_name in enumerate(tqdm(class_names, desc="Classes")):
            image_paths = images_by_class[class_name]

            # 如果指定了最大采样数，随机采样
            if max_samples_per_class and len(image_paths) > max_samples_per_class:
                np.random.seed(42)  # 保证可重复性
                image_paths = np.random.choice(image_paths, max_samples_per_class, replace=False).tolist()

            # 批量处理
            batch_size = 32
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                batch_images = []

                for img_path in batch_paths:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = preprocess(img)
                        batch_images.append(img_tensor)
                    except Exception as e:
                        print(f"Warning: Failed to load {img_path}: {e}")
                        continue

                if not batch_images:
                    continue

                batch_tensor = torch.stack(batch_images).to(device)
                features = model.encode_image(batch_tensor)
                features = F.normalize(features, dim=-1)  # L2归一化

                all_features.append(features.cpu().numpy())
                all_labels.extend([class_idx] * len(batch_images))

    features = np.vstack(all_features)
    labels = np.array(all_labels)

    print(f"Extracted {features.shape[0]} features with dimension {features.shape[1]}")

    return features, labels, class_names


def apply_tsne(features, perplexity=30, max_iter=1000, random_state=42):
    """应用t-SNE降维"""
    print(f"Applying t-SNE with perplexity={perplexity}...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
        n_jobs=-1  # 使用所有CPU核心
    )

    embeddings = tsne.fit_transform(features)

    return embeddings


def plot_tsne(embeddings, labels, class_names, dataset_name, output_path, title=None):
    """
    绘制t-SNE可视化图（IEEE论文风格）
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # 设置颜色循环
    n_classes = len(class_names)
    colors = IEEE_COLORS[:n_classes] if n_classes <= len(IEEE_COLORS) else plt.cm.tab20(np.linspace(0, 1, n_classes))

    # 为每个类别绘制散点
    for i, class_name in enumerate(class_names):
        mask = labels == i
        class_embeddings = embeddings[mask]

        # 使用不同的marker形状（每5个类别换一个形状）
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
        marker = markers[i % len(markers)]

        ax.scatter(
            class_embeddings[:, 0],
            class_embeddings[:, 1],
            c=colors[i] if n_classes <= len(IEEE_COLORS) else colors[i],
            marker=marker,
            s=50,  # 点的大小
            alpha=0.7,
            edgecolors='white',
            # linewidths=0.5,
            label=class_name,
            zorder=2
        )

    # # 添加图例（仅保留类别标签）
    # legend = ax.legend(
    #     loc='best',
    #     fontsize=9,
    #     frameon=True,
    #     fancybox=False,
    #     shadow=False,
    #     ncol=2 if n_classes > 8 else 1,
    #     title='Classes',
    #     title_fontsize=10,
    #     edgecolor='black'
    # )
    # legend.get_frame().set_linewidth(1.0)
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_alpha(0.95)

    # 移除坐标轴刻度、刻度值、标题和边框
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 移除边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved visualization to: {output_path}")

    plt.close()


def plot_tsne_with_density(embeddings, labels, class_names, dataset_name, output_path, title=None):
    """
    绘制带密度背景的t-SNE可视化图（更高级的版本）
    """
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # 设置颜色
    n_classes = len(class_names)
    colors = IEEE_COLORS[:n_classes] if n_classes <= len(IEEE_COLORS) else plt.cm.tab20(np.linspace(0, 1, n_classes))

    # 绘制密度背景
    if len(embeddings) > 100:
        try:
            kde = gaussian_kde(embeddings.T)
            x_min, x_max = embeddings[:, 0].min() - 1, embeddings[:, 0].max() + 1
            y_min, y_max = embeddings[:, 1].min() - 1, embeddings[:, 1].max() + 1

            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density = kde(positions).reshape(xx.shape)

            ax.contourf(xx, yy, density, levels=20, cmap='Greys', alpha=0.3, zorder=0)
        except:
            pass  # 如果密度计算失败，跳过

    # 为每个类别绘制散点
    for i, class_name in enumerate(class_names):
        mask = labels == i
        class_embeddings = embeddings[mask]

        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
        marker = markers[i % len(markers)]

        ax.scatter(
            class_embeddings[:, 0],
            class_embeddings[:, 1],
            c=colors[i] if n_classes <= len(IEEE_COLORS) else colors[i],
            marker=marker,
            s=60,
            alpha=0.8,
            edgecolors='white',
            linewidths=0.8,
            label=class_name,
            zorder=2
        )

    # 添加图例（仅保留类别标签）
    legend = ax.legend(
        loc='best',
        fontsize=9,
        frameon=True,
        fancybox=False,
        shadow=False,
        ncol=2 if n_classes > 8 else 1,
        title='Classes',
        title_fontsize=10,
        edgecolor='black'
    )
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)

    # 移除坐标轴刻度、刻度值、标题和边框
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 移除边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved visualization to: {output_path}")

    plt.close()


def visualize_dataset(
    model,
    preprocess,
    device,
    dataset_key,
    output_prefix,
    output_dir,
    max_samples_per_class=None,
    perplexity=30,
    use_density=False,
    custom_title=None
):
    """可视化单个数据集"""
    config = DATASET_CONFIGS[dataset_key]

    print(f"\n{'='*60}")
    print(f"Processing {config['name']}...")
    print(f"{'='*60}")

    # 获取图像
    images_by_class = get_images_from_folder(config['path'], config['extensions'])
    print(f"Found {len(images_by_class)} classes")

    for class_name, images in images_by_class.items():
        print(f"  {class_name}: {len(images)} images")

    # 提取特征
    features, labels, class_names = extract_features(
        model, preprocess, device, images_by_class, max_samples_per_class
    )

    # t-SNE降维
    embeddings = apply_tsne(features, perplexity=perplexity)

    # 绘制
    output_path = os.path.join(output_dir, f"{output_prefix}_{dataset_key}_tsne.png")

    title = custom_title if custom_title else f'{config["name"]} ({len(class_names)} classes)'

    if use_density:
        plot_tsne_with_density(embeddings, labels, class_names, config['name'], output_path, title)
    else:
        plot_tsne(embeddings, labels, class_names, config['name'], output_path, title)

    return embeddings, labels, class_names


def main():
    parser = argparse.ArgumentParser(
        description='t-SNE Visualization for SAR-VLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 可视化所有数据集（默认）
  python tsne_visualization.py --checkpoint /path/to/epoch_8.pt --model ViT-L-14

  # 只可视化MSTAR和FUSAR
  python tsne_visualization.py --checkpoint /path/to/epoch_8.pt --model ViT-L-14 --datasets mstar fusar

  # 指定输出前缀
  python tsne_visualization.py --checkpoint /path/to/epoch_8.pt --model ViT-L-14 --prefix "transferclip"

  # 限制每类样本数（加快大数据集处理）
  python tsne_visualization.py --checkpoint /path/to/epoch_8.pt --model ViT-L-14 --max-samples 100

  # 使用密度背景
  python tsne_visualization.py --checkpoint /path/to/epoch_8.pt --model ViT-L-14 --density
        """
    )

    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., /path/to/epoch_8.pt)')
    parser.add_argument('--model', type=str, default='ViT-L-14',
                        help='Model architecture name (default: ViT-L-14)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained model to load before checkpoint (optional)')

    # 数据集参数
    parser.add_argument('--datasets', nargs='+', choices=['mstar', 'fusar', 'sar_vsa'],
                        default=['mstar', 'fusar', 'sar_vsa'],
                        help='Datasets to visualize (default: all)')

    # 输出参数
    parser.add_argument('--prefix', type=str, default='model',
                        help='Output file prefix (default: model)')
    parser.add_argument('--output-dir', type=str, default='/home/maqw/SARVLM/SARVLM/vis',
                        help='Output directory for visualizations')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom title for plots')

    # t-SNE参数
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity (default: 30)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per class (default: no limit)')

    # 可视化参数
    parser.add_argument('--density', action='store_true',
                        help='Add density background to plots')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available)')

    args = parser.parse_args()

    # 检查输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 检查checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 加载模型
    print(f"\nLoading model: {args.model}")
    if args.pretrained:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model,
            pretrained=args.pretrained,
            device=device
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model,
            device=device
        )

    # 加载checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 处理不同格式的checkpoint
    # 检查是否为 TorchScript 模型
    if isinstance(checkpoint, torch.jit.ScriptModule):
        print("Detected TorchScript model, loading directly...")
        model = checkpoint
        model = model.to(device)
        model.eval()
    else:
        # 处理普通 state_dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # 移除 'module.' 前缀（如果是分布式训练的模型）
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict, strict=False)
        else:
            # 直接加载模型
            model = checkpoint

        model = model.to(device)
        model.eval()

    print("Model loaded successfully!")

    # 可视化每个数据集
    results = {}
    for dataset_key in args.datasets:
        embeddings, labels, class_names = visualize_dataset(
            model=model,
            preprocess=preprocess,
            device=device,
            dataset_key=dataset_key,
            output_prefix=args.prefix,
            output_dir=args.output_dir,
            max_samples_per_class=args.max_samples,
            perplexity=args.perplexity,
            use_density=args.density,
            custom_title=args.title
        )
        results[dataset_key] = {
            'embeddings': embeddings,
            'labels': labels,
            'class_names': class_names
        }

    print(f"\n{'='*60}")
    print("Visualization completed!")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for dataset_key in args.datasets:
        filename = f"{args.prefix}_{dataset_key}_tsne.png"
        print(f"  - {filename}")
    print(f"{'='*60}\n")

    return results


if __name__ == '__main__':
    main()
