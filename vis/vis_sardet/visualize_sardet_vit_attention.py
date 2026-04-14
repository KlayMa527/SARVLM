#!/usr/bin/env python3
"""
Visualize SARDet-100k val images with visual encoder attention maps only.

Attention map source:
- ViT visual self-attention rollout from CLS token over patch tokens.
- No text feature / no image-text similarity is used.

Saved outputs per image under --out-dir (default: vis_sardet/fig_vit):
1) original/
2) boxed/
3) attention/
4) overlay/
5) panel/
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import open_clip  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--images-dir",
        type=str,
        default="path/to/SARDet-100k/Images/val",
    )
    p.add_argument(
        "--ann-json",
        type=str,
        default="path/to/SARDet-100k/Annotations/val.json",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="path/to/SARCLIP-RemoteCLIP-ViT-L-14.pt",
    )
    p.add_argument("--model", type=str, default="ViT-L-14")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out-dir",
        type=str,
        default="path/to/fig_vit_SARCLIP-RemoteCLIP",
    )
    p.add_argument("--overlay-alpha", type=float, default=0.55)
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="0 means process all images with annotations in val.json.",
    )
    return p.parse_args()


def patch_grid_size(model, input_hw: tuple[int, int]) -> tuple[int, int]:
    ps = model.visual.patch_size
    if isinstance(ps, (tuple, list)):
        ph, pw = int(ps[0]), int(ps[1])
    else:
        ph = pw = int(ps)
    h, w = input_hw
    return h // ph, w // pw


def norm_map_for_viz(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def draw_bboxes(
    pil_img: Image.Image,
    anns: list[dict],
    selected_cat_id: int,
    color_sel=(255, 0, 0),
    color_other=(255, 255, 0),
) -> Image.Image:
    boxed = pil_img.copy()
    draw = ImageDraw.Draw(boxed)
    for ann in anns:
        x, y, w, h = ann["bbox"]
        xyxy = (x, y, x + w, y + h)
        if int(ann["category_id"]) == int(selected_cat_id):
            draw.rectangle(xyxy, outline=color_sel, width=2)
        else:
            draw.rectangle(xyxy, outline=color_other, width=1)
    return boxed


@torch.no_grad()
def vit_cls_attention_rollout(model, image_tensor: torch.Tensor, gh: int, gw: int) -> np.ndarray:
    """
    Compute CLS-token attention rollout heatmap for ViT visual encoder.
    Only supports visual transformer blocks with nn.MultiheadAttention.
    """
    visual = model.visual

    # Build token sequence: [CLS + patches]
    x = visual._embeds(image_tensor)  # [B, L, C], batch_first=True
    B, L, _ = x.shape
    if B != 1:
        raise RuntimeError("This function expects batch size = 1.")

    attn_mats = []
    for blk in visual.transformer.resblocks:
        # This script targets default ResidualAttentionBlock (nn.MultiheadAttention).
        if not isinstance(blk.attn, torch.nn.MultiheadAttention):
            raise RuntimeError(
                "Detected non-MultiheadAttention block in visual encoder. "
                "This script currently supports default ViT attention blocks only."
            )

        x_ln = blk.ln_1(x)
        # attn_w: [B, heads, L, L]
        attn_out, attn_w = blk.attn(
            x_ln,
            x_ln,
            x_ln,
            need_weights=True,
            average_attn_weights=False,
            attn_mask=None,
        )
        x = x + blk.ls_1(attn_out)
        x = x + blk.ls_2(blk.mlp(blk.ln_2(x)))

        attn_w = attn_w[0].float()  # [heads, L, L]
        attn_mean = attn_w.mean(dim=0)  # [L, L]
        # Add residual and row-normalize (attention rollout)
        eye = torch.eye(L, device=attn_mean.device, dtype=attn_mean.dtype)
        attn_mean = attn_mean + eye
        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        attn_mats.append(attn_mean)

    # Rollout = A_1 @ A_2 @ ... @ A_n
    rollout = attn_mats[0]
    for m in attn_mats[1:]:
        rollout = rollout @ m

    # CLS attention to patch tokens
    cls_to_patch = rollout[0, 1:]  # [L-1]
    if cls_to_patch.numel() != gh * gw:
        raise RuntimeError(
            f"Patch count mismatch in attention rollout: got {cls_to_patch.numel()}, expected {gh * gw}"
        )

    heat = cls_to_patch.reshape(gh, gw).cpu().numpy()
    return heat


def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    ann_json = Path(args.ann_json)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)

    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    if not ann_json.exists():
        raise FileNotFoundError(ann_json)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    dir_original = out_dir / "original"
    dir_boxed = out_dir / "boxed"
    dir_attention = out_dir / "attention"
    dir_overlay = out_dir / "overlay"
    dir_panel = out_dir / "panel"
    for d in (dir_original, dir_boxed, dir_attention, dir_overlay, dir_panel):
        d.mkdir(parents=True, exist_ok=True)

    with open(ann_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    categories = coco.get("categories", [])
    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in categories}

    img_id_to_anns = defaultdict(list)
    for a in anns:
        img_id_to_anns[int(a["image_id"])].append(a)

    valid_images = [im for im in images if len(img_id_to_anns[int(im["id"])]) > 0]
    valid_images.sort(key=lambda x: int(x["id"]))
    if args.max_images > 0:
        valid_images = valid_images[: args.max_images]

    device = torch.device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model,
        pretrained=None,
        device=device,
    )
    open_clip.load_checkpoint(
        model=model,
        checkpoint_path=str(ckpt_path),
        strict=False,
        weights_only=True,
        device=device,
    )
    model.eval()
    model_dtype = next(model.parameters()).dtype

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        cmap_fn = matplotlib.colormaps["turbo"]
    except Exception:
        cmap_fn = plt.cm.turbo

    total = len(valid_images)
    processed = 0
    skipped = 0

    for idx, im in enumerate(valid_images):
        img_id = int(im["id"])
        file_name = str(im["file_name"])
        img_path = images_dir / file_name
        ann_list = img_id_to_anns[img_id]

        if not img_path.exists():
            skipped += 1
            continue

        try:
            pil_raw = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        cat_counter = Counter(int(a["category_id"]) for a in ann_list)
        selected_cat_id = cat_counter.most_common(1)[0][0]
        selected_cat_name = cat_id_to_name.get(selected_cat_id, f"class_{selected_cat_id}")
        present_names = sorted(
            {cat_id_to_name.get(int(a["category_id"]), str(int(a["category_id"]))) for a in ann_list}
        )

        image_tensor = preprocess(pil_raw).unsqueeze(0).to(device=device)
        if model_dtype in (torch.float16, torch.bfloat16):
            image_tensor = image_tensor.to(dtype=model_dtype)
        _, _, th, tw = image_tensor.shape
        gh, gw = patch_grid_size(model, (th, tw))

        # Display image aligned with model input
        pp_cfg = open_clip.get_model_preprocess_cfg(model)
        mean = torch.tensor(pp_cfg.get("mean", (0.48145466, 0.4578275, 0.40821073)), dtype=torch.float32)
        std = torch.tensor(pp_cfg.get("std", (0.26862954, 0.26130258, 0.27577711)), dtype=torch.float32)
        mean = mean.view(3, 1, 1).to(image_tensor.device)
        std = std.view(3, 1, 1).to(image_tensor.device)
        vis_chw = (image_tensor[0].float() * std + mean).clamp(0, 1).cpu().numpy()
        vis_hwc = (vis_chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        orig_display = Image.fromarray(vis_hwc)

        # Scale annotation boxes to display resolution
        scale_x = tw / float(im["width"])
        scale_y = th / float(im["height"])
        ann_scaled = []
        for a in ann_list:
            x, y, w, h = a["bbox"]
            ann_scaled.append(
                {
                    "category_id": int(a["category_id"]),
                    "bbox": [x * scale_x, y * scale_y, w * scale_x, h * scale_y],
                }
            )
        boxed = draw_bboxes(orig_display, ann_scaled, selected_cat_id=selected_cat_id)

        try:
            heat_raw = vit_cls_attention_rollout(model, image_tensor, gh, gw)
        except Exception as e:
            print(f"[skip] {file_name}: {e}")
            skipped += 1
            continue

        heat_norm = norm_map_for_viz(heat_raw)
        colored = np.asarray(cmap_fn(heat_norm))[..., :3]
        colored_u8 = (colored * 255.0).astype(np.uint8)
        heat_pil = Image.fromarray(colored_u8).resize((tw, th), Image.BILINEAR)

        orig_arr = np.asarray(orig_display).astype(np.float32) / 255.0
        heat_arr = np.asarray(heat_pil).astype(np.float32) / 255.0
        overlay_arr = (1.0 - args.overlay_alpha) * orig_arr + args.overlay_alpha * heat_arr
        overlay_u8 = np.clip(overlay_arr * 255.0, 0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_u8)

        stem = f"{idx:05d}_{Path(file_name).stem}"
        orig_display.save(dir_original / f"{stem}.png")
        boxed.save(dir_boxed / f"{stem}.png")
        heat_pil.save(dir_attention / f"{stem}.png")
        overlay_pil.save(dir_overlay / f"{stem}.png")

        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7))
        axes[0].imshow(orig_display)
        axes[0].set_title("Original", fontsize=10)
        axes[0].axis("off")
        axes[1].imshow(boxed)
        axes[1].set_title("With boxes", fontsize=10)
        axes[1].axis("off")
        axes[2].imshow(overlay_pil)
        axes[2].set_title("ViT attention (overlay)", fontsize=10)
        axes[2].axis("off")

        info = (
            f"Selected category: {selected_cat_name} | Present categories: {', '.join(present_names)}\n"
            "Attention: CLS rollout over visual transformer self-attention"
        )
        fig.text(0.5, 0.01, textwrap.fill(info, width=140), ha="center", va="bottom", fontsize=8)
        plt.tight_layout(rect=[0, 0.13, 1, 0.98])
        fig.savefig(dir_panel / f"{stem}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        processed += 1
        if processed % 200 == 0:
            print(f"Processed {processed}/{total}")

    print(
        f"[OK] finished. processed={processed}, skipped={skipped}, total_listed={total}. "
        f"Saved in {out_dir}/{{original,boxed,attention,overlay,panel}}"
    )


if __name__ == "__main__":
    main()

