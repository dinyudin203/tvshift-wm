#!/usr/bin/env bash
# Pixel-axis motion reasoning on pair5 (H100, GPUs 4-7): 5 pooled same-viewpoint context clips, front camera only,
# no calibrated ray input. MODE=molmoact|ki  CM=5|10  UNITS=chunk_step|context_step|total|cm
#   MODE=ki CM=5 GPUS=0 bash run_motion.sh
set -euo pipefail
R=/dataset/yujin/vla/icwm_reason
PY=/dataset/yujin/envs/icwm/bin/python
MODE=${MODE:?molmoact or ki}; CM=${CM:?5 or 10}; UNITS=${UNITS:-chunk_step}
export HF_HOME=/dataset/yujin/hf_home WANDB_DIR=$R/wandb
export PYTHONPATH=$R/code/src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export ICWM_PREP_IN_WORKERS=1 ICWM_SPARSE_LOGITS=1
export ICWM_GC_SKIP_LAYERS=${ICWM_GC_SKIP_LAYERS:-6}  # keep activations of the first 6 LM layers (H100 80GB headroom)
export ICWM_FUSED_ADAMW=1 ICWM_DDP_BUCKET_VIEW=1
export ICWM_VISION_FLASH_ATTN=${ICWM_VISION_FLASH_ATTN-kernels-community/flash-attn2}
export ICWM_FRAME_CACHE=${ICWM_FRAME_CACHE-/dev/shm/yujin_frame_cache_pair5}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPUS:-4,5,6,7}
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
STEPS=${STEPS:-12000}; BS=${BS:-16}
NAME=icwm_pair5_ctx5_motion${MODE}_${CM}cm_${UNITS}
OUT=${OUT:-$R/runs/${NAME}_b$((BS*NPROC))_s$STEPS}
D=/dataset/yujin/data/tvshift-libero-pair5-lerobot
ROOTS="[$D/libero_spatial_theta000_no_noops_lerobot,$D/libero_goal_theta015_no_noops_lerobot,$D/libero_object_theta030_no_noops_lerobot,$D/libero_10_theta330_no_noops_lerobot,$D/libero_10_theta345_no_noops_lerobot]"
FIRST=$D/libero_spatial_theta000_no_noops_lerobot
HUB=$HF_HOME/hub
QWEN=$(ls -d $HUB/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/*/ | head -1)
FAST=$(ls -d $HUB/models--lerobot--fast-action-tokenizer/snapshots/*/ | head -1)
LAUNCH=(--mixed_precision=bf16 --num_processes="$NPROC" --main_process_port="${PORT:-29700}")
[ "$NPROC" -gt 1 ] && LAUNCH=(--multi_gpu "${LAUNCH[@]}")
cd $R/code
train() {
/dataset/yujin/envs/icwm/bin/accelerate launch "${LAUNCH[@]}" run_icwm_train.py \
  --policy.type=icwm --policy.backbone_name_or_path="$QWEN" --policy.fast_tokenizer_name_or_path="$FAST" \
  --policy.n_context_clips=5 --policy.n_history_clips=0 --policy.context_source=pool \
  --policy.context_match_viewpoint=true --policy.context_pool_dataset_roots="$ROOTS" \
  --policy.normalize_context_actions=true \
  '--policy.context_image_keys=[observation.images.image]' \
  '--policy.drop_image_keys=[observation.images.wrist_image]' \
  --policy.motion_reasoning=true --policy.motion_reasoning_mode="$MODE" \
  --policy.motion_reasoning_cm="$CM" --policy.motion_reasoning_units="$UNITS" \
  --policy.plucker_frame=base --policy.plucker_calibration_dir=/dataset/yujin/vla/icwm_plucker_tv6/calibration_pair5 \
  --policy.action_expert_lr="${EXPERT_LR:-1e-4}" \
  --policy.action_chunk_size=5 --policy.n_action_steps=5 \
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 \
  --policy.optimizer_lr=5e-5 --policy.optimizer_weight_decay=1e-4 \
  --policy.scheduler_warmup_steps="${WARMUP:-2000}" --policy.scheduler_decay_steps="$STEPS" --policy.push_to_hub=false \
  --dataset.repo_id="$FIRST" --dataset.root="$FIRST" --dataset.video_backend=pyav \
  --batch_size="$BS" --steps="$STEPS" --num_workers="${NW:-6}" \
  --log_freq=20 --save_freq="${SAVE_FREQ:-2000}" --eval_freq=0 --seed=1000 \
  --output_dir="$OUT" --job_name="$NAME" \
  --wandb.enable="${WANDB:-true}" --wandb.project=tvshift-vla --wandb.entity=AdaptWam \
  --wandb.mode=online --wandb.disable_artifact=true "$@"
}
set +e +o pipefail
TMP=$(mktemp /tmp/motion_attempt.XXXX.log)
train "$@" 2>&1 | tee "$TMP"; rc=${PIPESTATUS[0]}
# OOM with the larger no-recompute set before any checkpoint: retry once with the original setting (1 layer)
if [ "$rc" -ne 0 ] && [ "$ICWM_GC_SKIP_LAYERS" -gt 1 ] && grep -q "OutOfMemoryError\|CUDA out of memory" "$TMP" && [ ! -d "$OUT/checkpoints" ]; then
  echo "=== OOM with ICWM_GC_SKIP_LAYERS=$ICWM_GC_SKIP_LAYERS; retrying with 1 ==="
  mv "$OUT" "${OUT}.oom_$(date +%s)" 2>/dev/null
  export ICWM_GC_SKIP_LAYERS=1
  train "$@"; rc=$?
fi
rm -f "$TMP"; exit $rc
