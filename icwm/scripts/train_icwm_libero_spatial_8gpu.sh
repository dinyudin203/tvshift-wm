#!/usr/bin/env bash
# ICWM training launch, LIBERO-Spatial, 8x RTX A6000 (49GB each), bf16, DDP via `accelerate`.
#
# Hyperparameters per the paper (Sec 5 / Table 5): AdamW, weight_decay=1e-4, lr peak 5e-5,
# 50k-step warmup then cosine decay, action_chunk_size=5, n_context_clips=5. Real ICWM
# in-context conditioning is wired via `--policy.context_pool_dataset_roots` (see
# src/lerobot/datasets/factory.py::_maybe_wrap_icwm_context_sampler and
# src/lerobot/datasets/icwm_context_sampler.py), pooling interaction clips across all 8
# in-domain training azimuths (30/60/90/120/240/270/300/330), per paper Sec 4.1 -- this
# was NOT wired into lerobot_train.py before this run; see icwm/README.md.
#
# `run_icwm_train.py` wraps `lerobot.scripts.lerobot_train` and disables cuDNN: this
# machine's driver (535.274.02 / CUDA 12.2) + torch==2.5.1+cu121's bundled cuDNN 9.2.0
# combination fails on Qwen2.5-VL's vision patch-embed conv3d with
# CUDNN_STATUS_NOT_INITIALIZED; disabling cuDNN (only cost: that one small conv op no longer
# uses cuDNN kernels; every other op in this model is attention/matmul, unaffected) fixes it.

set -euo pipefail

REPO_ROOT="/workspace/yujin/icwm/lerobot"
DATASET_ROOT="/workspace/yujin/data/libero_spatial_rot/lerobot_v30"
VENV="/workspace/yujin/icwm/.venv_gpu"

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"
export HF_HOME=/workspace/yujin/icwm/.hf_home
export PYTHONPATH=src

ROOTS="[${DATASET_ROOT}/libero_spatial_theta030_no_noops_lerobot,${DATASET_ROOT}/libero_spatial_theta060_no_noops_lerobot,${DATASET_ROOT}/libero_spatial_theta090_no_noops_lerobot,${DATASET_ROOT}/libero_spatial_theta120_no_noops_lerobot,${DATASET_ROOT}/libero_spatial_theta240_no_noops_lerobot,${DATASET_ROOT}/libero_spatial_theta270_no_noops_lerobot,${DATASET_ROOT}/libero_spatial_theta300_no_noops_lerobot,${DATASET_ROOT}/libero_spatial_theta330_no_noops_lerobot]"

accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  --mixed_precision=bf16 \
  run_icwm_train.py \
  --policy.type=icwm \
  --policy.backbone_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct" \
  --policy.n_context_clips=5 \
  --policy.action_chunk_size=5 \
  --policy.n_action_steps=5 \
  --policy.fast_tokenizer_name_or_path="lerobot/fast-action-tokenizer" \
  --policy.context_pool_dataset_roots="${ROOTS}" \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.scheduler_warmup_steps=50000 \
  --policy.scheduler_decay_steps=300000 \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET_ROOT}/libero_spatial_theta030_no_noops_lerobot" \
  --batch_size=2 \
  --steps=300000 \
  --log_freq=20 \
  --save_freq=2000 \
  --eval_freq=0 \
  --num_workers=2 \
  --output_dir="/workspace/yujin/icwm/outputs/train/icwm_libero_spatial_table3" \
  --wandb.enable=true \
  --wandb.project="icwm-table3" \
  --wandb.mode=online \
  --wandb.disable_artifact=true \
  --resume=true \
  --job_name="icwm_libero_spatial_table3"
