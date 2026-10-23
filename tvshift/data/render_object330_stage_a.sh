#!/bin/bash
# Stage A rendering for the corrected setup: Object task0,task1 @330 (replaces Spatial@330)
set -u
RES=${RES:-512}
PY=/workspace/envs/fastwam/bin/python
SCRIPT=/workspace/yujin/libero_rot/render_shard.py
BASE_OUT=/workspace/yujin/data/fastwam_setupshift

run_task() {
  local suite=$1 raw=$2 task=$3 thetas=$4
  local out="${BASE_OUT}/${suite}/shards"
  local logdir="${BASE_OUT}/${suite}/logs"
  mkdir -p "$out" "$logdir"
  for chunk in 0 1 2; do
    local start=$(( chunk * 17 ))
    local end=$(( start + 17 ))
    [ $end -gt 50 ] && end=50
    [ $start -ge 50 ] && continue
    MUJOCO_GL=egl PYTHONPATH=/workspace/yujin/LIBERO-JY \
      "$PY" "$SCRIPT" --suite "$suite" --raw-dir "$raw" \
        --task "$task" --demo-start "$start" --demo-end "$end" \
        --thetas $thetas --res "$RES" --out "$out" \
        > "$logdir/task${task}_chunk${chunk}.log" 2>&1 &
  done
}

run_task libero_object /workspace/yujin/data/libero_raw/libero_object 0 "330"
run_task libero_object /workspace/yujin/data/libero_raw/libero_object 1 "330"

wait
echo "STAGE A COMPLETE (object330)"
