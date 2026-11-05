#!/bin/bash
# v2wrist: tvshift v2 (front camera 3 viewpoints) + wrist(eye_in_hand) camera perturbation.
#   front   0 + wrist   0 : spatial 1,3 (= tvshift v2 spatial theta000, reused as a symlink)
#   front  45 + wrist +30 : goal    0,7
#   front 330 + wrist -30 : object  0,1
# Stage A: states replay of the 50 raw demos per task (render_shard.py --wrist-deg: eye_in_hand rotated about the
# gripper approach axis = robot0_right_hand z), 3 shards per task on 4 GPUs. Stage B: pack_lerobot.py per suite.
# HF: dinyu/tvshift-libero-v2wrist-lerobot
set -uo pipefail
B=${B:-/dataset/yujin/data/v2wrist}
RAW=${RAW:-/dataset/yujin/data/libero-raw-hdf5}
V2=${V2:-/dataset/yujin/data/tvshift-libero-lerobot}
PY=${PY:-/dataset/yujin/envs/tv/bin/python}
C=$(cd "$(dirname "$0")" && pwd)
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-/dataset/yujin/libero_cfg} PYTHONPATH=${LIBERO_ROOT:-/home/gpuuser/yujin/repos/LIBERO}
mkdir -p $B/shards $B/logs $B/lerobot_v30
i=0
for cfg in "libero_goal 0 45 30" "libero_goal 7 45 30" "libero_object 0 330 -30" "libero_object 1 330 -30"; do
  set -- $cfg
  for r in "0 17" "17 34" "34 50"; do
    set -- $1 $2 $3 $4 $r; g=$((i % 4)); i=$((i + 1))
    CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g nice -n 10 $PY $C/render_shard.py --suite $1 --raw-dir $RAW/$1 \
      --task $2 --demo-start $5 --demo-end $6 --thetas $3 --wrist-deg=$4 --out $B/shards/$1 > $B/logs/shard_$1_t$2_$5.log 2>&1 &
  done
done
wait
echo ALL SHARDS DONE
$PY $C/pack_lerobot.py --shards $B/shards/libero_goal --theta 45 --suite libero_goal --out $B/lerobot_v30 \
  --name libero_goal_theta045_wrist030_no_noops_lerobot > $B/logs/pack_goal.log 2>&1 || { echo "PACK FAILED goal"; exit 1; }
$PY $C/pack_lerobot.py --shards $B/shards/libero_object --theta 330 --suite libero_object --out $B/lerobot_v30 \
  --name libero_object_theta330_wristm030_no_noops_lerobot > $B/logs/pack_object.log 2>&1 || { echo "PACK FAILED object"; exit 1; }
ln -sfn $V2/libero_spatial_theta000_no_noops_lerobot $B/lerobot_v30/libero_spatial_theta000_wrist000_no_noops_lerobot
echo ALL PACKED
