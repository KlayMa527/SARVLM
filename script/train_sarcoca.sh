

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 配置路径
PRETRAINED_PATH="path/to//sarcoca_pretrain_hqrs_210k/checkpoints/epoch_10.pt"
TRAIN_DATA="path/to/train_acsv"
VAL_DATA="path/to/val_uni_image_and_caption.csv"
LOGS_DIR="./logs"
EXPERIMENT_NAME="sarcoca_from_openclip_hqrs_sarvlm"

# 运行训练
python -m open_clip_train.main \
    --dataset-type "csv" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --csv-img-key "imgpath" \
    --csv-caption-key "caption" \
    --csv-separator "," \
    --model "coca_ViT-L-14" \
    --pretrained "${PRETRAINED_PATH}" \
    --epochs 10 \
    --batch-size 64 \
    --lr 1e-5 \
    --wd 0.1 \
    --warmup 1000 \
    --workers 8 \
    --precision "amp" \
    --save-frequency 1 \
    --report-to "tensorboard" \
    --logs "${LOGS_DIR}" \
    --name "${EXPERIMENT_NAME}" \
    --coca-contrastive-loss-weight 1.0 \
    --coca-caption-loss-weight 1.0 \
    --log-every-n-steps 100 \
    --seed 42

