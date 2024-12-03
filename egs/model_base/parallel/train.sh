#!/bin/bash

# call with train.sh nametag model_name
set -eo pipefail

echo "[Train] Whisper-CLIP Parallel Base on Flickr8k"

BASE_DIR=/compute/babel-11-13/sbharad2/m3l_runs
# timestamp=$(date "+%Y%m%d.%H%M%S")

mynametag=${1:-"exp"}
model_name=${2:-"TrainKWClip_SpeechText"}

EXP_ROOT="${BASE_DIR}/${mynametag}"
TRAINING_CFG="config/speechCLIP/model_base/whispclp_p.yaml"

mkdir -p $EXP_ROOT

python3 run_task.py "${model_name}" \
    --config $TRAINING_CFG \
    --gpus 1 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT
