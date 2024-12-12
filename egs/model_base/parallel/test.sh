set -ex

echo "[Test] Whisper-CLIP Parallel on Flickr8k"

# EXP_NAME="img_mean.contd.train_lin.3636020"
# # CKPT_NAME="epoch=166-step=19538-val_loss=1.4593.ckpt"
# CKPT_NAME="epoch=168-step=19772-val_recall_mean_10=71.7407.ckpt"
# model_name="TrainImgMean_SpeechText"

# Baseline
# EXP_NAME="t.linear_proj.20241201.133909"
# CKPT_NAME="epoch=208-step=24452-val_recall_mean_10=69.9097.ckpt"
# model_name="TrainKWClip_SpeechText"
# model_name="TrainNormalizedSpeechImg_SpeechText"

# EVAL_CFG="/data/user_data/sbharad2/SpeechCLIP/config/speechCLIP/model_base/whispclip_p_inference.yaml"
##--

# EXP_NAME="txt_mean.spclip.train_lin.3639175"
# CKPT_NAME="epoch=117-step=13805-val_recall_mean_10=89.1634.ckpt"
# model_name="TrainTextMean_KWClip_GeneralTransformer"
# EVAL_CFG="/data/user_data/sbharad2/SpeechCLIP/config/speechCLIP/model_base/spchclp_p.yaml"

# Large, sp-txt transitive
EXP_NAME="contd.txt_mean.spc_large.3642524.3643366"
CKPT_NAME="epoch=45-step=5381-val_recall_mean_10=63.8650.ckpt"
model_name="TrainTextMean_KWClip_GeneralTransformer"
EVAL_CFG="/data/user_data/sbharad2/SpeechCLIP/config/speechCLIP/model_large/flickr/spchclp_oriented.yaml"

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
