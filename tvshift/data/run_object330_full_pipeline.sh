#!/bin/bash
set -euo pipefail
BASE=/workspace/yujin/data/fastwam_setupshift
PY_FASTWAM=/workspace/envs/fastwam/bin/python
PY_DP=/workspace/envs/dp_setupshift/bin/python
PY_ROLLOUT=/workspace/envs/dp_rollout/bin/python
DPDIR=/workspace/yujin/diffusion_policy_idm

echo "### STAGE B: pack object330 ###"
bash /workspace/yujin/libero_rot/stage_b_pack_object330.sh

echo "### split_json for object330 ###"
cd /workspace/yujin/libero_rot
"$PY_DP" make_split_json.py "$BASE/lerobot_v30/setupshift_libero_object_theta330" \
  setupshift_libero_object_theta330 "$BASE/splits/setupshift_libero_object_theta330_split.json"

echo "### precompute_cache for object330 ###"
cd "$DPDIR"
PYTHONPATH="$DPDIR" "$PY_DP" -m idm.precompute_cache \
  --root "$BASE/lerobot_v30/setupshift_libero_object_theta330" \
  --repo_id setupshift_libero_object_theta330 \
  --out "$BASE/cache/setupshift_libero_object_theta330_84" \
  --size 84 --workers 16 --batch_size 128

echo "### TRAIN v2 (0=Spatial1,3 / 15=Goal0,7 / 330=Object0,1 / 345=Goal5,2) ###"
cd "$DPDIR"
PYTHONPATH="$DPDIR" CUDA_VISIBLE_DEVICES=0 "$PY_DP" -m idm.train_setupshift_dp --wandb

echo "### EVAL: seen + unseen pairs ###"
CKPT="$DPDIR/runs/setupshift_dp_transformer_v2/latest.pt"
OUT_ROOT="$DPDIR/runs/setupshift_dp_transformer_v2_eval"
mkdir -p "$OUT_ROOT"
cd "$DPDIR/idm"

run_combo() {
  local group=$1 suite=$2 task=$3 theta=$4
  echo "=== $group $suite task$task theta$theta ==="
  FASTWAM_LIBERO_CAMERA_THETA=$theta FASTWAM_LIBERO_IMAGE_FLIP=v FASTWAM_LIBERO_GRIPPER=raw \
    PYTHONPATH=/workspace/yujin/LIBERO-JY:$DPDIR:/workspace/yujin/FastWAM/experiments/libero \
    CUDA_VISIBLE_DEVICES=0 "$PY_ROLLOUT" rollout_setupshift_dp.py \
    --ckpt "$CKPT" --suite "$suite" --task "$task" --num-trials 5 \
    --out "$OUT_ROOT/${group}_${suite}_task${task}_theta${theta}.csv"
}

# ---- seen (trained) pairs, 8 ----
run_combo seen libero_spatial 1 0
run_combo seen libero_spatial 3 0
run_combo seen libero_goal 0 15
run_combo seen libero_goal 7 15
run_combo seen libero_object 0 330
run_combo seen libero_object 1 330
run_combo seen libero_goal 5 345
run_combo seen libero_goal 2 345

# ---- unseen pairs: task seen (own theta) x theta seen (via another task), combo untrained. 24 ----
run_combo unseen libero_spatial 1 15
run_combo unseen libero_spatial 1 330
run_combo unseen libero_spatial 1 345
run_combo unseen libero_spatial 3 15
run_combo unseen libero_spatial 3 330
run_combo unseen libero_spatial 3 345

run_combo unseen libero_goal 0 0
run_combo unseen libero_goal 0 330
run_combo unseen libero_goal 0 345
run_combo unseen libero_goal 7 0
run_combo unseen libero_goal 7 330
run_combo unseen libero_goal 7 345
run_combo unseen libero_goal 5 0
run_combo unseen libero_goal 5 15
run_combo unseen libero_goal 5 330
run_combo unseen libero_goal 2 0
run_combo unseen libero_goal 2 15
run_combo unseen libero_goal 2 330

run_combo unseen libero_object 0 0
run_combo unseen libero_object 0 15
run_combo unseen libero_object 0 345
run_combo unseen libero_object 1 0
run_combo unseen libero_object 1 15
run_combo unseen libero_object 1 345

echo "OBJECT330_FULL_PIPELINE_DONE"
