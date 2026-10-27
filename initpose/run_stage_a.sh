#!/bin/bash
# Stage A: training initial-pose conditions. One shard dir (= one dataset) per condition,
# 5 workers per task over its 50 demos.
cd "$(dirname "$0")"
PY=/opt/conda/envs/lingbot-va/bin/python
OUT=/workspace/yujin/data/libero_initpose
mkdir -p $OUT/logs
CONDS=(  # suite task mag(signed rad, 0 = default) seed
  "libero_spatial 1 0 0"
  "libero_spatial 3 0 0"
  "libero_goal 0 0.2 0"
  "libero_goal 7 0.2 0"
  "libero_object 0 -0.3 0"
  "libero_object 1 -0.3 0"
)
for c in "${CONDS[@]}"; do
  read suite task mag seed <<< "$c"
  name=${suite}_initpose_$(awk "BEGIN{m=$mag; if (m == 0) print \"default\"; else printf \"dq%s%03d\", (m > 0 ? \"p\" : \"m\"), (m > 0 ? m : -m) * 100}")
  for w in 0 1 2 3 4; do
    s=$((w * 10)); e=$((s + 10)); pv=$([ $w -eq 0 ] && echo 1 || echo 0)
    MUJOCO_GL=egl $PY make_shard.py --suite $suite --task $task --mag $mag --seed $seed \
      --demo-start $s --demo-end $e --preview $pv --out $OUT/shards/$name > $OUT/logs/${name}_task${task}_w$w.log 2>&1 &
  done
done
wait
