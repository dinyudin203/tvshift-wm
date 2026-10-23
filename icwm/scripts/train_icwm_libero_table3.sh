#!/usr/bin/env bash
# ICWM training launch script for the Table 3 protocol (LIBERO-Spatial/Goal/Object/Long,
# 8 in-domain training azimuths). NOT EXECUTED as part of this deliverable: all 8 GPUs on
# this machine are currently at 100% utilization from another user's job (see
# icwm/README.md), and the 3 non-Spatial suites are not yet rendered at any azimuth (see
# scripts/render_libero_cross_view.sh).
#
# Hyperparameters follow the paper (Sec 5 / Table 5): AdamW, weight_decay=1e-4, lr peak
# 5e-5, 50k-step linear warmup then cosine decay, action_chunk_size=5, n_context_clips=5,
# backbone=Qwen2.5-VL-3B, 8x A100 (informational only -- not launched here).

set -euo pipefail

REPO_ROOT="/workspace/yujin/icwm/lerobot"
DATASET_ROOT="/workspace/yujin/data/libero_spatial_rot/lerobot_v30"  # Spatial-only until
                                                                       # the other 3 suites
                                                                       # are rendered.

cd "${REPO_ROOT}"

python src/lerobot/scripts/lerobot_train.py \
  --policy.type=icwm \
  --policy.backbone_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct" \
  --policy.n_context_clips=5 \
  --policy.action_chunk_size=5 \
  --policy.n_action_steps=5 \
  --policy.fast_tokenizer_name_or_path="lerobot/fast-action-tokenizer" \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.scheduler_warmup_steps=50000 \
  --policy.gradient_checkpointing=true \
  --env.type=libero \
  --env.task_suite_name=libero_spatial \
  --dataset.repo_id="${DATASET_ROOT}/libero_spatial_theta030_no_noops_lerobot" \
  --batch_size=8 \
  --steps=300000 \
  --output_dir="/workspace/yujin/icwm/outputs/train/icwm_libero_spatial_table3" \
  --wandb.enable=false \
  --job_name="icwm_libero_spatial_table3"

# TODO once all 4 suites are rendered/converted (scripts/render_libero_cross_view.sh):
#   - Replace --dataset.repo_id with a multi-dataset spec spanning all 8 training azimuths
#     x the target suite (see lerobot.datasets.multi_dataset.MultiLeRobotDataset), and wrap
#     it with lerobot.datasets.icwm_context_sampler.ICWMContextSampledDataset so every
#     training item gets N context clips drawn from the cross-azimuth interaction-segment
#     pool (build via icwm_context_sampler.build_icwm_context_pool_from_azimuth_datasets).
#   - Repeat for libero_object / libero_goal / libero_10, each as a separate run (Table 3
#     reports per-suite success rates).
