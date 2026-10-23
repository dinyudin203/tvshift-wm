#!/bin/bash
# Stage B for the setup-shift subset: stage only the needed (task,theta) episodes
# into a scratch shards dir (symlinks), then run pack_lerobot.py per combo.
set -euo pipefail
PY=/workspace/envs/fastwam/bin/python
BASE=/workspace/yujin/data/fastwam_setupshift
OUT="${BASE}/lerobot_v30"
mkdir -p "$OUT"

stage_and_pack() {
  local suite=$1 theta=$2 tasks=$3   # tasks = space-separated 0-padded task ids e.g. "01 03"
  local theta_tag=$(printf "%03d" "$theta")
  local src="${BASE}/${suite}/shards"
  local stage="${BASE}/${suite}/stage_theta${theta_tag}"
  rm -rf "$stage"
  mkdir -p "$stage/_done" "$stage/frames" "$stage/wrist" "$stage/theta${theta_tag}"

  for t in $tasks; do
    for f in "$src/_done/task${t}_"*.json; do
      [ -e "$f" ] || continue
      stem=$(basename "$f" .json)
      ln -sf "$src/_done/${stem}.json"        "$stage/_done/${stem}.json"
      ln -sf "$src/frames/${stem}.npz"        "$stage/frames/${stem}.npz"
      ln -sf "$src/wrist/${stem}.mp4"         "$stage/wrist/${stem}.mp4"
      ln -sf "$src/theta${theta_tag}/${stem}.mp4" "$stage/theta${theta_tag}/${stem}.mp4"
    done
  done

  n=$(ls "$stage/_done" | wc -l)
  echo "=== ${suite} theta=${theta} tasks=[${tasks}] -> staged ${n} episodes ==="
  "$PY" /workspace/yujin/libero_rot/pack_lerobot.py \
    --shards "$stage" --theta "$theta" --suite "$suite" --out "$OUT" \
    --name "setupshift_${suite}_theta${theta_tag}"
}

stage_and_pack libero_spatial 0   "01 03"
stage_and_pack libero_spatial 330 "00 01"
stage_and_pack libero_goal    15  "00 07"
stage_and_pack libero_goal    345 "02 05"

echo "STAGE B COMPLETE"
