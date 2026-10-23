#!/usr/bin/env bash
# Orchestration script for Table 3's cross-view protocol: render LIBERO-Object,
# LIBERO-Goal and LIBERO-10 (Long) at the 8 in-domain training azimuths (30, 60,
# 90, 120, 240, 270, 300, 330 degrees) and convert each suite/azimuth pair to a
# LeRobot v3.0 dataset, using the SAME pipeline that produced
# /workspace/yujin/data/libero_spatial_rot/.
#
# ============================================================================
# PIPELINE FOUND (this is the actual answer to "how was libero_spatial_rot
# produced" -- it does NOT live under LIBERO-JY/xyg_scripts/, which only has a
# frame-preview tool and a hardcoded exploration script; the real two-stage
# renderer + LeRobot-v3.0 packer lives at):
#
#   /workspace/yujin/libero_rot/render_shard.py   (Stage A: full-episode render)
#   /workspace/yujin/libero_rot/pack_lerobot.py   (Stage B: LeRobot v3.0 packer)
#   /workspace/yujin/libero_rot/run_stage_a.sh, run_stage_b.sh, chain_stage_b.sh
#   /workspace/yujin/libero_rot/README.md         (full pipeline writeup, Korean)
#
# (also mirrored, byte-identical, at /workspace/yujin/FastWAM/libero_rot/).
# Both render_shard.py and pack_lerobot.py already accept --suite/--raw-dir, so
# no code changes were needed to generalize beyond libero_spatial -- only the
# raw hdf5 source data and output paths differ per suite. This script is that
# generalization.
#
# Stage A replays each demo's simulator `states` frame-by-frame
# (`sim.set_state_from_flattened`) and re-renders the `agentview` camera at
# each theta after rotating it about the robot-base z-axis (identical math to
# LIBERO-JY/xyg_scripts/only_rotate_dataset.py's rotate_camera_based_on_robot_base,
# verified numerically consistent in libero_rot/README.md). Since this replays
# recorded state (not actions), there is no rollout drift across thetas. Stage B
# packs the per-episode per-theta mp4s + npz state/action arrays into one
# LeRobot v3.0 dataset per (suite, theta), matching the exact schema used by
# libero_spatial_rot (README: yuanty/LIBERO-fastwam schema).
#
# Raw hdf5 source data (downloaded via LIBERO's own
# libero.libero.utils.download_utils.libero_dataset_download, same official
# box.com links LIBERO-JY/benchmark_scripts/download_libero_datasets.py uses):
#   libero_object -> /workspace/yujin/data/libero_raw/libero_object/  (10 hdf5, 50 demos each)
#   libero_goal   -> /workspace/yujin/data/libero_raw/libero_goal/    (10 hdf5, 50 demos each)
#   libero_10     -> /workspace/yujin/data/libero_raw/libero_10/      (already present, 10 hdf5, 50 demos each)
#
# Env: /workspace/envs/fastwam (CPU-only container -- no NVIDIA EGL/GL, so
# MuJoCo offscreen rendering runs on llvmpipe software rendering, ~4-5
# renders/s per worker; with ~30 parallel workers a full suite (10 tasks x 50
# demos x 8 output thetas + 1 shared wrist render) takes roughly the same
# 1.5-2h wall clock that libero_spatial_rot's README reports for the analogous
# 9-theta spatial run).
#
# Output layout mirrors libero_spatial_rot/lerobot_v30/ exactly:
#   /workspace/yujin/data/<suite>_rot/shards/          (Stage A intermediate: per-episode mp4/npz)
#   /workspace/yujin/data/<suite>_rot/lerobot_v30/<suite>_theta<NNN>_no_noops_lerobot/
#   /workspace/yujin/data/<suite>_rot/logs/            (per-task-chunk + per-theta-pack logs)
#   /workspace/yujin/data/<suite>_rot/stage_a.log, stage_b.log
#
# libero_10 (aka "libero_long") datasets are named libero_10_theta<NNN>_no_noops_lerobot
# (i.e. the internal LIBERO benchmark name, matching pack_lerobot.py's --suite
# default naming convention) under /workspace/yujin/data/libero_10_rot/.
# ============================================================================

set -euo pipefail

LIBERO_ROT="/workspace/yujin/libero_rot"
LIBERO_JY_ROOT="/workspace/yujin/LIBERO-JY"
PY=/workspace/envs/fastwam/bin/python
AZIMUTHS=(30 60 90 120 240 270 300 330)

declare -A RAW_DIRS=(
  [libero_object]="/workspace/yujin/data/libero_raw/libero_object"
  [libero_goal]="/workspace/yujin/data/libero_raw/libero_goal"
  [libero_10]="/workspace/yujin/data/libero_raw/libero_10"
)

echo "== libero_spatial =="
echo "Already rendered (8 azimuths + theta000) at:"
echo "  /workspace/yujin/data/libero_spatial_rot/lerobot_v30/"
echo "Skipping re-render."

for suite in libero_object libero_goal libero_10; do
  raw_dir="${RAW_DIRS[$suite]}"
  out_root="/workspace/yujin/data/${suite}_rot"
  shards="${out_root}/shards"
  lerobot_out="${out_root}/lerobot_v30"
  logdir="${out_root}/logs"
  mkdir -p "${shards}" "${lerobot_out}" "${logdir}"

  echo "== ${suite} =="
  echo "  raw hdf5: ${raw_dir}"
  echo "  out: ${out_root}"

  n_hdf5=$(find "${raw_dir}" -maxdepth 1 -iname '*.hdf5' | wc -l)
  if [[ "${n_hdf5}" -ne 10 ]]; then
    echo "  !! expected 10 hdf5 files in ${raw_dir}, found ${n_hdf5} -- skipping ${suite}" >&2
    continue
  fi

  # ---- Stage A: 10 tasks x 3 demo chunks (17/17/16) = 30 workers, all thetas
  # rendered per demo (wrist rendered once, shared across thetas).
  for task in 0 1 2 3 4 5 6 7 8 9; do
    for chunk in 0 1 2; do
      start=$(( chunk * 17 ))
      end=$(( start + 17 ))
      [ $end -gt 50 ] && end=50
      [ $start -ge 50 ] && continue
      MUJOCO_GL=egl PYTHONPATH="${LIBERO_JY_ROOT}" \
        "$PY" "${LIBERO_ROT}/render_shard.py" \
          --suite "${suite}" --raw-dir "${raw_dir}" \
          --task "$task" --demo-start "$start" --demo-end "$end" \
          --thetas "${AZIMUTHS[@]}" \
          --res 512 --out "${shards}" \
          > "${logdir}/task${task}_chunk${chunk}.log" 2>&1 &
    done
  done
  wait
  echo "STAGE A COMPLETE" > "${out_root}/stage_a.log"
  n_done=$(ls "${shards}/_done" 2>/dev/null | wc -l)
  echo "  stage A finished with ${n_done} episodes"

  # ---- Stage B: one LeRobot v3.0 dataset per theta, packed in parallel.
  for theta in "${AZIMUTHS[@]}"; do
    "$PY" "${LIBERO_ROT}/pack_lerobot.py" \
      --shards "${shards}" --out "${lerobot_out}" --theta "${theta}" --suite "${suite}" \
      > "${logdir}/pack_theta${theta}.log" 2>&1 &
  done
  wait
  echo "STAGE B COMPLETE" >> "${out_root}/stage_b.log"
  echo "  done -> ${lerobot_out}"
done

echo ""
echo "All suites processed. Per-suite logs under /workspace/yujin/data/<suite>_rot/logs/."
