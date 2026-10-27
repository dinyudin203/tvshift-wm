#!/bin/bash
# extra retries (20) for the demos still missing in Goal 0 / Goal 7; previews of the last attempt are kept
cd "$(dirname "$0")"
PY=/opt/conda/envs/lingbot-va/bin/python
OUT=/workspace/yujin/data/libero_initpose
for d in 1 12 23 27 31 32 40 43 45; do
  MUJOCO_GL=egl $PY make_shard.py --suite libero_goal --task 0 --mag 0.2 --seed 0 --demo-start $d --demo-end $((d+1)) \
    --retries 20 --max-track-mm 20 --preview 1 --out $OUT/shards/libero_goal_initpose_dqp020 > $OUT/logs/retry2_goal0_d$d.log 2>&1 &
done
MUJOCO_GL=egl $PY make_shard.py --suite libero_goal --task 7 --mag 0.2 --seed 0 --demo-start 24 --demo-end 25 \
  --retries 20 --max-track-mm 20 --preview 1 --out $OUT/shards/libero_goal_initpose_dqp020 > $OUT/logs/retry2_goal7_d24.log 2>&1 &
wait
