#!/usr/bin/env bash
# ICWM with Reflective-VLA style context (context_source=history: the 5 chunks just before the
# query in its own episode, gap U[0,15]); everything else = train_icwm_libero_tvshift_v2_8gpu.sh.
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
DATASET_ROOT="/workspace/yujin/data/libero_tvshift/lerobot_v30"
VENV="/workspace/yujin/icwm/.venv_gpu"

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"
export HF_HOME=/workspace/yujin/icwm/.hf_home
export PYTHONPATH=src
# 8 ranks on 31 cores: the default of one intra-op thread per core makes every CPU op ~40x slower
# (e.g. clamp/mul/round on a 512x512 image: 127 ms at 31 threads vs 3 ms at 2)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# vision encoder window attention as one FlashAttention-2 varlen call per layer instead of one
# SDPA call per window (same attention; measured -31% vision fwd+bwd, bf16 error vs fp32 unchanged)
export ICWM_VISION_FLASH_ATTN=1
export ICWM_FRAME_CACHE=/workspace/yujin/data/icwm_tvshift_cache

ROOTS="[${DATASET_ROOT}/libero_spatial_theta000_no_noops_lerobot,${DATASET_ROOT}/libero_goal_theta045_no_noops_lerobot,${DATASET_ROOT}/libero_goal_theta345_no_noops_lerobot,${DATASET_ROOT}/libero_object_theta270_no_noops_lerobot]"

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
  --policy.context_match_viewpoint=true \
  --policy.context_source=history \
  --policy.context_history_max_gap=15 \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.scheduler_warmup_steps=${WARMUP:-1000} \
  --policy.scheduler_decay_steps=${STEPS:-20000} \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET_ROOT}/libero_spatial_theta000_no_noops_lerobot" \
  --batch_size=2 \
  --steps=${STEPS:-20000} \
  --log_freq=20 \
  --save_freq=2500 \
  --eval_freq=0 \
  --num_workers=2 \
  --output_dir="/workspace/yujin/icwm/outputs/train/icwm_libero_tvshift_hist" \
  --wandb.enable=true \
  --wandb.project="icwm-tvshift" \
  --wandb.mode=online \
  --wandb.disable_artifact=true \
  --job_name="icwm_libero_tvshift_hist"
