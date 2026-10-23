#!/usr/bin/env bash
# Resume ICWM training on LIBERO-Spatial from the last checkpoint.
# Root cause of the crash at step 2000: WandB artifact name exceeded 128 chars
# (long absolute dataset path baked into the run "group" string). Fixed at the
# code level (name now truncated in wandb_utils.get_safe_wandb_artifact_name)
# AND disabled here via --wandb.disable_artifact=true as belt-and-suspenders.
set -euo pipefail

REPO_ROOT="/workspace/yujin/icwm/lerobot"
VENV="/workspace/yujin/icwm/.venv_gpu"
OUTPUT_DIR="/workspace/yujin/icwm/outputs/train/icwm_libero_tvshift_mixdrop"
CONFIG_PATH="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"

cd "${REPO_ROOT}"
source "${VENV}/bin/activate"
export HF_HOME=/workspace/yujin/icwm/.hf_home
export PYTHONPATH=src
# 8 ranks on 31 cores: the default of one intra-op thread per core makes every CPU op ~40x slower
# (e.g. clamp/mul/round on a 512x512 image: 127 ms at 31 threads vs 3 ms at 2)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# 10 context clips leave little headroom on 49GB: avoid fragmentation OOMs (8GB reserved-but-free at the first OOM)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vision encoder window attention as one FlashAttention-2 varlen call per layer instead of one
# SDPA call per window (same attention; measured -31% vision fwd+bwd, bf16 error vs fp32 unchanged)
export ICWM_VISION_FLASH_ATTN=1
# decode every frame once (build_icwm_cache.py) instead of seeking videos ~36x per sample
export ICWM_FRAME_CACHE=/workspace/yujin/data/icwm_tvshift_cache
# torch.profiler over 10 steps on the first run after the cache switch (tables land in <output>/profile)
# (no torch.profiler for v2; profiled on v1)

accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  --mixed_precision=bf16 \
  run_icwm_train.py \
  --config_path="${CONFIG_PATH}" \
  --resume=true \
  --wandb.disable_artifact=true
