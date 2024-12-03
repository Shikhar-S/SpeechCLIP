set -ex

echo "[Test] Whisper-CLIP Parallel Base on Flickr8k"

EXP_NAME="img_mean.train_lin.3631259"
CKPT_NAME="epoch=120-step=14156-val_recall_mean_10=70.8130.ckpt"
model_name="TrainImgMean_SpeechText"

# Baseline
# EXP_NAME="t.linear_proj.20241201.133909"
# CKPT_NAME="epoch=208-step=24452-val_recall_mean_10=69.9097.ckpt"
# model_name="TrainKWClip_SpeechText"

EVAL_CFG="/data/user_data/sbharad2/SpeechCLIP/config/speechCLIP/model_base/whispclip_p_inference.yaml"

BASE_DIR=/compute/babel-11-13/sbharad2/m3l_runs
EVAL_CKPT="${BASE_DIR}/${EXP_NAME}/${CKPT_NAME}"


# DUMMY
EXP_ROOT="/data/user_data/sbharad2/SpeechCLIP/exp_test/inference"
mkdir -p $EXP_ROOT

python3 run_task.py "${model_name}" \
    --config "${EVAL_CFG}" \
    --resume "${EVAL_CKPT}" \
    --config "${EVAL_CFG}" \
    --gpus 1 \
    --seed 7122 \
    --test \
    --save_path $EXP_ROOT
