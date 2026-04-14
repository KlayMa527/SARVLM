

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

PRETRAINED_PATH="path/to/remote-clip-ViT-14.pt"
TRAIN_DATA="path/to/train_csv"
VAL_DATA="path/to/val_csv"
LOGS_DIR="path/to/logs"
EXPERIMENT_NAME="SARCLIP-RemoteCLIP-ViT-L-14"

torchrun --nproc_per_node=8 --master_port=29310  -m open_clip_train.main \
  --train-data "${TRAIN_DATA}" \
  --val-data "${VAL_DATA}" \
  --dataset-type csv \
  --csv-separator "," \
  --csv-img-key imgpath \
  --csv-caption-key caption \
  --dataset-type csv \
  --model ViT-L-14 \
  --pretrained "${PRETRAINED_PATH}" \
  --lr=5e-5 \
  --lr-scheduler cosine \
  --warmup 1000 \
  --batch-size 128 \
  --epochs 10 \
  --precision amp \
  --wd 0.01 \
  --workers 12 \
  --grad-clip-norm 1.0 \
  --logs "${LOGS_DIR}" \
  --name "${EXPERIMENT_NAME}" \