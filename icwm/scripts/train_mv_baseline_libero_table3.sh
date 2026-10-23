#!/usr/bin/env bash
# MV baseline: multi-view behavior cloning WITHOUT in-context conditioning, i.e. the exact
# same ICWMPolicy/backbone with n_context_clips=0 (no interaction clips T prepended --
# reduces to a standard pi_theta(a_t | o_t, l) VLA trained across the same 8 viewpoints).
# NOT EXECUTED (see icwm/README.md and train_icwm_libero_table3.sh).

set -euo pipefail

REPO_ROOT="/workspace/yujin/icwm/lerobot"
cd "${REPO_ROOT}"

python src/lerobot/scripts/lerobot_train.py \
  --policy.type=icwm \
  --policy.backbone_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct" \
  --policy.n_context_clips=0 \
  --policy.action_chunk_size=5 \
  --policy.n_action_steps=5 \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.scheduler_warmup_steps=50000 \
  --env.type=libero \
  --env.task_suite_name=libero_spatial \
  --dataset.repo_id="/workspace/yujin/data/libero_spatial_rot/lerobot_v30/libero_spatial_theta030_no_noops_lerobot" \
  --batch_size=8 \
  --steps=300000 \
  --output_dir="/workspace/yujin/icwm/outputs/train/mv_baseline_libero_spatial_table3" \
  --wandb.enable=false \
  --job_name="mv_baseline_libero_spatial_table3"
