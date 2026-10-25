#!/usr/bin/env bash
# ICWM backbone w/o context (n_context_clips=0): Qwen2.5-VL-3B-Instruct + FAST action tokens, tvshift merged data,
# global batch 16 x 8 GPU = 128, 6000 steps. Recipe otherwise as scripts/train_icwm_libero_tvshift_v2_8gpu.sh.
set -uo pipefail
OUT=${OUT:-/dataset/yujin/vla/runs/icwm_noctx_b128_s6000}
export HF_HOME=/dataset/yujin/hf_home WANDB_DIR=/dataset/yujin/wandb OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
cd /home/gpuuser/yujin/repos/icwm/icwm/lerobot
exec /dataset/yujin/envs/icwm/bin/accelerate launch --multi_gpu --num_processes=8 --mixed_precision=bf16 --main_process_port=29582 \
  run_icwm_train.py \
  --policy.type=icwm --policy.backbone_name_or_path=Qwen/Qwen2.5-VL-3B-Instruct \
  --policy.n_context_clips=0 --policy.action_chunk_size=5 --policy.n_action_steps=5 \
  --policy.fast_tokenizer_name_or_path=lerobot/fast-action-tokenizer \
  --policy.gradient_checkpointing=false --policy.dtype=bfloat16 \
  --policy.optimizer_lr=5e-5 --policy.optimizer_weight_decay=1e-4 \
  --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=6000 --policy.push_to_hub=false \
  --dataset.repo_id=local/tvshift_train4 --dataset.root=/dataset/yujin/data/tvshift_train4_lerobot --dataset.video_backend=pyav \
  --batch_size=16 --steps=6000 --log_freq=20 --save_freq=1000 --eval_freq=0 --num_workers=4 \
  --output_dir=$OUT --job_name=icwm_noctx_tvshift \
  --wandb.enable=true --wandb.project=tvshift-vla --wandb.entity=AdaptWam --wandb.mode=online --wandb.disable_artifact=true "$@"
