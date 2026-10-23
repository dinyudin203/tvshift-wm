#!/bin/bash
# Stage B: theta 별 LeRobot v3.0 데이터셋 패킹 (9개 병렬)
set -u
SHARDS=${SHARDS:-/workspace/yujin/data/libero_spatial_rot/shards}
OUT=${OUT:-/workspace/yujin/data/libero_spatial_rot/lerobot_v30}
LOGDIR=${LOGDIR:-/workspace/yujin/data/libero_spatial_rot/logs}
PY=/workspace/envs/fastwam/bin/python
mkdir -p "$OUT" "$LOGDIR"

for theta in 0 30 60 90 120 240 270 300 330; do
  "$PY" /workspace/yujin/libero_rot/pack_lerobot.py \
    --shards "$SHARDS" --out "$OUT" --theta "$theta" \
    > "$LOGDIR/pack_theta${theta}.log" 2>&1 &
done
wait
echo "STAGE B COMPLETE"
