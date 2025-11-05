#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

ROOT_PATH="<YOUR PATH>"
DATA_PATH="<YOUR PATH>"

DATA_NAME="SteeringAngle"
MIN_LABEL=-80.0
MAX_LABEL=80.0
IMG_SIZE=64

python auxiliary_regression.py \
    --root_path "$ROOT_PATH" --data_path "$DATA_PATH" --data_name "$DATA_NAME" \
    --min_label "$MIN_LABEL" --max_label "$MAX_LABEL" --img_size "$IMG_SIZE" \
    --net_name resnet18 --epochs 200 \
    --batch_size_train 256 --base_lr 0.01 --weight_dacay 1e-4 \
    --use_amp --mixed_precision_type fp16 \
    2>&1 | tee output_${DATA_NAME}_PreAuxReg.txt