#!/bin/bash
# Re-run the Goal demos that were rejected or had large post-merge tracking error.
# Already finished demos are skipped (_done markers); the rest retry up to 5x with the same q0.
cd "$(dirname "$0")"
PY=/opt/conda/envs/lingbot-va/bin/python
OUT=/workspace/yujin/data/libero_initpose
name=libero_goal_initpose_dqp020
for task in 0 7; do
  for w in 0 1 2 3 4 5 6 7 8 9; do
    s=$((w * 5)); e=$((s + 5))
    MUJOCO_GL=egl $PY make_shard.py --suite libero_goal --task $task --mag 0.2 --seed 0 \
      --demo-start $s --demo-end $e --preview 0 --retries 5 --max-track-mm 20 \
      --out $OUT/shards/$name > $OUT/logs/${name}_retry_task${task}_w$w.log 2>&1 &
  done
done
wait
