cd path/to/SARVLM
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0

# openclip
python eval/RET/eval_retrieval.py \
    --model ViT-L-14 \
    --pretrained path/to/openaiCLIP/ViT-L-14.pt \
    --checkpoint path/to/openaiCLIP/ViT-L-14.pt \
    --data-csv path/to/val_uni_image_and_caption_sample_5000.csv \
    --batch-size 32 \
    --device cuda \
    --output-dir path/to//output \
    --output-name openclip_ViT-L-14_5000

# remoteclip
python eval/RET/eval_retrieval.py \
    --model ViT-L-14 \
    --pretrained path/to/openaiCLIP/ViT-L-14.pt \
    --checkpoint path/to/RemoteCLIP-ViT-L-14.pt \
    --data-csv path/to/val_uni_image_and_caption_sample_5000.csv \
    --batch-size 32 \
    --device cuda \
    --output-dir path/to//output \
    --output-name remoteclip_ViT-L-14_5000

# georsclip
python eval/RET/eval_retrieval.py \
    --model ViT-L-14 \
    --pretrained path/to/openaiCLIP/ViT-L-14.pt \
    --checkpoint path/to/GEORSCLIP/RS5M_ViT-L-14.pt \
    --data-csv path/to/val_uni_image_and_caption_sample_5000.csv \
    --batch-size 32 \
    --device cuda \
    --output-dir path/to//output \
    --output-name georsclip_ViT-L-14_5000

# skyclip
python eval/RET/eval_retrieval.py \
    --model ViT-L-14 \
    --pretrained path/to/openaiCLIP/ViT-L-14.pt \
    --checkpoint path/to/skyCLIP/SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt \
    --data-csv path/to/val_uni_image_and_caption_sample_5000.csv \
    --batch-size 32 \
    --device cuda \
    --output-dir path/to//output \
    --output-name skyclip_ViT-L-14_5000



# SARCLIP-Remote
python eval/RET/eval_retrieval.py \
    --model ViT-L-14 \
    --pretrained path/to/SARCLIP-Remote-ViT-L-14.pt \
    --checkpoint path/to/skyCLIP/SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt \
    --data-csv path/to/val_uni_image_and_caption_sample_5000.csv \
    --batch-size 32 \
    --device cuda \
    --output-dir path/to//output \
    --output-name skyclip_ViT-L-14_5000