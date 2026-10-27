#!/bin/bash
# Stage B: pack each initial-pose condition into its own LeRobot v3.0 dataset
# with FastWAM/libero_rot/pack_lerobot.py (agentview clips live in theta000/).
PY=/opt/conda/envs/lingbot-va/bin/python
OUT=/workspace/yujin/data/libero_initpose
for d in $OUT/shards/*/; do
  name=$(basename $d)
  $PY /workspace/yujin/FastWAM/libero_rot/pack_lerobot.py --shards $d --out $OUT/lerobot_v30 \
    --theta 0 --res 512 --name ${name}_no_noops_lerobot > $OUT/logs/pack_${name}.log 2>&1 &
done
wait
