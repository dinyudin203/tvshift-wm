#!/usr/bin/env bash
# ICWM with TWO context segments on tvshift, total context length 5 clips = the single-source runs (explore-only / history-only):
# "Exploration:" NE=3 self-exploration probing clips + "History:" NH=2 chunks of the query's own episode (dropped with prob HDROP=0.3).
# ICWM WITH context on tvshift: same data (the 4 tvshift training datasets = tvshift_train4, 393 eps, 8 tasks) and
# recipe as train_icwm.sh (no-context tvshift run: b16 x 8 = 128, 6000 steps, lr 5e-5, warmup 1000), plus
# n_context_clips=5, context_source=pool, viewpoint-matched (per paper; = scripts/train_icwm_libero_tvshift_v2_8gpu.sh),
# and the speed/memory fixes of train_icwm_ctx_libero4.sh (GC with first 6 layers skipped, sparse logits, frame cache).
# CONTEXT=explore (default): the clips are SELF-EXPLORATION probing clips (build_icwm_explore_context.py: the eval's
# probe() run at the query's viewpoint, in a scene of the query's task, init states 10..49), not demo segments.
# CONTEXT=pool: demo segments of other episodes of the same viewpoint (ICWM paper training).
# Context-clip actions are normalized with the target action stats before FAST tokenization (CTX_NORM=false: raw, as in libero4 ctx).
set -uo pipefail
STEPS=${STEPS:-6000}
CONTEXT=${CONTEXT:-explore}
EXPLORE=${EXPLORE:-/dev/shm/icwm_tvshift_explore}
D=/dataset/yujin/data/tvshift-libero-lerobot
OUT=${OUT:-/dataset/yujin/vla/runs/icwm_ctxboth_e3h2_tvshift_b128_s${STEPS}}
ROOTS="[$D/libero_spatial_theta000_no_noops_lerobot,$D/libero_goal_theta045_no_noops_lerobot,$D/libero_goal_theta345_no_noops_lerobot,$D/libero_object_theta270_no_noops_lerobot]"
export HF_HOME=/dataset/yujin/hf_home WANDB_DIR=/dataset/yujin/wandb OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src
export ICWM_VISION_FLASH_ATTN=kernels-community/flash-attn2 ICWM_PREP_IN_WORKERS=1 ICWM_SPARSE_LOGITS=1 ICWM_GC_SKIP_LAYERS=${GC_SKIP:-6} TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
export ICWM_FRAME_CACHE=/dev/shm/icwm_tvshift_cache
cd /home/gpuuser/yujin/repos/icwm/icwm/lerobot
exec /dataset/yujin/envs/icwm/bin/accelerate launch --multi_gpu --num_processes=8 --mixed_precision=bf16 --main_process_port=29587 \
  run_icwm_train.py \
  --policy.type=icwm --policy.backbone_name_or_path=/dataset/yujin/hf_home/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --policy.n_context_clips=${NE:-3} --policy.action_chunk_size=5 --policy.n_action_steps=5 \
  --policy.context_pool_dataset_roots="$ROOTS" --policy.context_match_viewpoint=true --policy.context_source=$CONTEXT --policy.context_explore_dir=$EXPLORE --policy.normalize_context_actions=${CTX_NORM:-true} --policy.n_history_clips=${NH:-2} --policy.history_dropout=${HDROP:-0.3} \
  --policy.fast_tokenizer_name_or_path=lerobot/fast-action-tokenizer \
  --policy.gradient_checkpointing=${GC:-true} --policy.dtype=bfloat16 \
  --policy.optimizer_lr=5e-5 --policy.optimizer_weight_decay=1e-4 \
  --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=${STEPS} --policy.push_to_hub=false \
  --dataset.repo_id=$D/libero_spatial_theta000_no_noops_lerobot --dataset.root=$D/libero_spatial_theta000_no_noops_lerobot --dataset.video_backend=pyav \
  --batch_size=${BS:-16} --steps=${STEPS} --log_freq=20 --save_freq=1000 --eval_freq=0 --num_workers=${NW:-10} \
  --output_dir=$OUT --job_name=icwm_ctxboth_e3h2_tvshift \
  --wandb.enable=${WANDB:-true} --wandb.project=tvshift-vla --wandb.entity=AdaptWam --wandb.mode=online --wandb.disable_artifact=true "$@"
