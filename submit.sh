#!/bin/bash

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=m3l
#SBATCH --partition=general
#SBATCH --output /compute/babel-11-13/sbharad2/m3l_runs/logs/%x-%J.txt
#SBATCH --gres=gpu:A6000:2
#SBATCH --time 2-00:00:00

cd /data/user_data/sbharad2/SpeechCLIP
source ~/.bashrc
conda activate whispclip
echo "Starting run: ${SLURM_JOB_NAME}.${SLURM_JOB_ID}"

./egs/model_base/parallel/train.sh "${SLURM_JOB_NAME}.${SLURM_JOB_ID}" $@
# ./egs/model_base/parallel/resume_training.sh "${SLURM_JOB_NAME}.${SLURM_JOB_ID}" $@
