#!/usr/bin/env bash
# Resume ICWM training on LIBERO-Spatial from the last checkpoint.
# Root cause of the crash at step 2000: WandB artifact name exceeded 128 chars
# (long absolute dataset path baked into the run "group" string). Fixed at the
# code level (name now truncated in wandb_utils.get_safe_wandb_artifact_name)
# AND disabled here via --wandb.disable_artifact=true as belt-and-suspenders.
set -euo pipefail

REPO_ROOT="/workspace/yujin/icwm/lerobot"
VENV="/workspace/yujin/icwm/.venv_gpu"
OUTPUT_DIR="/workspace/yujin/icwm/outputs/train/icwm_libero_spatial_table3"
CONFIG_PATH="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"
export HF_HOME=/workspace/yujin/icwm/.hf_home
export PYTHONPATH=src

accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  --mixed_precision=bf16 \
  run_icwm_train.py \
  --config_path="${CONFIG_PATH}" \
  --resume=true \
  --wandb.disable_artifact=true
