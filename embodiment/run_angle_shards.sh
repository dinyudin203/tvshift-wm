#!/bin/bash
# Arm + gripper + agentview angle data: IIWA+Robotiq85 0deg spatial 2,6 / UR5e+RG2 45deg goal 1,7 / Panda+Panda 270deg object 7,8.
# IIWA at 0deg is exactly the libero_embodiment_gripper run (same settings), so its shards are reused (symlinked); the other
# two are rendered here for all 50 demos, then N = smallest per-task success count, select + pack to LeRobot v3.0.
cd /workspace/yujin/embodiment_swap_v2
export MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=/workspace/yujin/data/libero_embodiment_gripper_angle
PY=/workspace/envs/fastwam/bin/python
RG="--gripper OnRobotRG2Gripper --menagerie_opt --tip_comp --grip_speed 0.01 --modes goal"
mkdir -p $OUT/shards logs
ln -sfn /workspace/yujin/data/libero_embodiment_gripper/shards/IIWA_Robotiq85_libero_spatial $OUT/shards/IIWA_Robotiq85_a0_libero_spatial
for cfg in "libero_goal 1 UR5e UR5e_RG2_a45 45" "libero_goal 7 UR5e UR5e_RG2_a45 45" \
           "libero_object 7 Panda Panda_PandaGripper_a270 270" "libero_object 8 Panda Panda_PandaGripper_a270 270"; do
  set -- $cfg
  case $4 in UR5e_RG2_a45) F=$RG;; *) F="";; esac
  for r in "0 25" "25 50"; do
    set -- $cfg $r
    nice -n 19 $PY render_emb_shard.py --suite $1 --task $2 --robot $3 --theta $5 --demo-start $6 --demo-end $7 $F \
      --out $OUT/shards/${4}_${1} > logs/ga_shard_${4}_${1}_t${2}_$6.log 2>&1 &
  done
done
wait
echo ALL SHARDS DONE
N=$(for d in IIWA_Robotiq85_a0_libero_spatial:2 IIWA_Robotiq85_a0_libero_spatial:6 UR5e_RG2_a45_libero_goal:1 UR5e_RG2_a45_libero_goal:7 \
             Panda_PandaGripper_a270_libero_object:7 Panda_PandaGripper_a270_libero_object:8; do
  ls $OUT/shards/${d%%:*}/_done/task$(printf %02d ${d##*:})_demo*.json | wc -l; done | sort -n | head -1)
echo "N=$N"
for dt in IIWA_Robotiq85_a0_libero_spatial:0 UR5e_RG2_a45_libero_goal:45 Panda_PandaGripper_a270_libero_object:270; do
  d=${dt%%:*}; th=${dt##*:}
  $PY select_shards.py --shards $OUT/shards/$d --out $OUT/shards_$N/$d --n $N --theta $th > logs/ga_select_$d.log 2>&1 || { echo "SELECT FAILED $d"; exit 1; }
  nice -n 19 $PY /workspace/yujin/FastWAM/libero_rot/pack_lerobot.py --shards $OUT/shards_$N/$d --out $OUT/lerobot_v30 --theta $th \
    --name ${d}_no_noops_lerobot > logs/ga_pack_$d.log 2>&1 || { echo "PACK FAILED $d"; exit 1; }
done
echo ALL PACKED
