#!/bin/bash
# Stage A 완료를 기다렸다가 Stage B 를 자동 실행한다.
set -u
until grep -q "STAGE A COMPLETE" /workspace/yujin/data/libero_spatial_rot/stage_a.log 2>/dev/null; do sleep 60; done
echo "stage A finished with $(ls /workspace/yujin/data/libero_spatial_rot/shards/_done | wc -l) episodes"
exec /workspace/yujin/libero_rot/run_stage_b.sh
