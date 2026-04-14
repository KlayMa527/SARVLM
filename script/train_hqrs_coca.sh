#!/bin/bash
# SARCoCA预训练脚本
# 使用CoCa预训练权重进行SARCoCA模型预训练

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 配置路径
PRETRAINED_PATH="path/to/laionCoCa-ViT-L-14-laion2B-s13B-b90k/open_clip_pytorch_model.bin"
TRAIN_DATA="path/to/HQRS-IT-210K_Dataset/1.26million_captions_a100.csv"
LOGS_DIR="path/logs"
EXPERIMENT_NAME="sarcoca_pretrain_hqrs_210k"

# 运行训练
python -m open_clip_train.main \
    --dataset-type "csv" \
    --train-data "${TRAIN_DATA}" \
    --csv-img-key "filepath" \
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

