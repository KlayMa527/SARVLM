#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/visualize_sardet_vit_attention.py" \
  --images-dir "path/to/SARDet-100k/Images/val" \
  --ann-json "path/to/SARDet-100k/Annotations/val.json" \
  --checkpoint "path/to/SARCLIP-RemoteCLIP-ViT-L-14.pt" \
  --model "ViT-L-14" \
  --out-dir "path/to/fig_vit_SARCLIP-RemoteCLIP"


python3 "${SCRIPT_DIR}/visualize_sardet_vit_attention.py" \
  --images-dir "path/to/SARDet-100k/Images/val" \
  --ann-json "path/to/SARDet-100k/Annotations/val.json" \
  --checkpoint "path/to/sarclip_isprs_vit_l_14_model.pth" \
  --model "ViT-L-14" \
  --out-dir "path/to/fig_vit_SARCLIP_ISPRS"

