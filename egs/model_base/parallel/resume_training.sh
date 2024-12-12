#!/bin/bash

# call with train.sh nametag model_name
set -eo pipefail

echo "[Train] Whisper-CLIP Parallel Base on Flickr8k"

BASE_DIR=/compute/babel-11-13/sbharad2/m3l_runs
# timestamp=$(date "+%Y%m%d.%H%M%S")

mynametag=${1:-"exp"}
model_name=${2:-"TrainKWClip_SpeechText"}

EXP_ROOT="${BASE_DIR}/${mynametag}"

TRAINING_CFG="config/speechCLIP/model_large/flickr/spchclp_oriented.yaml"
# RESUME_CKPT="${BASE_DIR}/img_mean.train_lin.3631259/epoch=166-step=19538-val_loss=1.4593.ckpt"
RESUME_CKPT="${BASE_DIR}/txt_mean.spc_large.3642524/epoch=15-step=1871-val_recall_mean_10=52.3150.ckpt"

mkdir -p $EXP_ROOT

python3 run_task.py "${model_name}" \
    --config $TRAINING_CFG \
    --resume "${RESUME_CKPT}" \
    --gpus 1 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT
