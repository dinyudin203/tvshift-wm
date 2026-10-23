#!/bin/bash
# Stage A: 10 tasks x 3 demo chunks = 30 workers
set -u
OUT=${OUT:-/workspace/yujin/data/libero_spatial_rot/shards}
LOGDIR=${LOGDIR:-/workspace/yujin/data/libero_spatial_rot/logs}
RES=${RES:-512}
PY=/workspace/envs/fastwam/bin/python
mkdir -p "$OUT" "$LOGDIR"

for task in 0 1 2 3 4 5 6 7 8 9; do
  for chunk in 0 1 2; do
    start=$(( chunk * 17 ))
    end=$(( start + 17 ))
    [ $end -gt 50 ] && end=50
    [ $start -ge 50 ] && continue
    MUJOCO_GL=egl PYTHONPATH=/workspace/yujin/LIBERO-JY \
      "$PY" /workspace/yujin/libero_rot/render_shard.py \
        --task "$task" --demo-start "$start" --demo-end "$end" \
        --res "$RES" --out "$OUT" \
        > "$LOGDIR/task${task}_chunk${chunk}.log" 2>&1 &
  done
done
wait
echo "STAGE A COMPLETE"
