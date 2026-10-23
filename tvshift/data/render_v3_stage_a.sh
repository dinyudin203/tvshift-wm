#!/bin/bash
# Stage A rendering for corrected angles: Goal task0,7 @45 (was 15), Object task0,1 @270 (was 330)
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
        > "$logdir/task${task}_chunk${chunk}_theta${thetas// /_}.log" 2>&1 &
  done
}

run_task libero_goal   /workspace/yujin/data/libero_raw/libero_goal   0 "45"
run_task libero_goal   /workspace/yujin/data/libero_raw/libero_goal   7 "45"
run_task libero_object /workspace/yujin/data/libero_raw/libero_object 0 "270"
run_task libero_object /workspace/yujin/data/libero_raw/libero_object 1 "270"

wait
echo "STAGE A COMPLETE (v3: goal45 + object270)"
