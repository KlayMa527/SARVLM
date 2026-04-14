python eval/zeroshot/eval_zeroshot.py \
    --model ViT-L-14 \
    --checkpoint path/to/SARCLIP-RemoteCLIP-ViT-L-14 \
    --data-root path/to/RSISC/NWPU-RESISC45/test \
    --templates sar_opt \
    --batch-size 32 \
    --device cuda \
    --output-dir path/to/output/NWPU-RESISC45 \
    --output-name sarclip_remoteclip_nwpu_resisc45_ViT-L-14


python eval/zeroshot/eval_zeroshot.py \
    --model ViT-L-14 \
    --checkpoint path/to/transferclip_wo_mstar/checkpoints/epoch.pt \
    --data-root path/to/MSTAR_SOC/TEST \
    --templates sar_opt \
    --batch-size 32 \
    --device cuda \
    --output-dir path/to/output/output/MSTAR \
    --output-name sarclip_remote/wo_mstar_mstar_ViT-L-14