#!/bin/bash
# Keep the first N successful demos per task (same N for every task) and pack to LeRobot v3.0, as run_embrq_shards.sh.
N=${1:?usage: pack_gripper.sh N}
cd /workspace/yujin/embodiment_swap_v2
OUT=/workspace/yujin/data/libero_embodiment_gripper
PY=/workspace/envs/fastwam/bin/python
for d in IIWA_Robotiq85_libero_spatial UR5e_RG2_libero_goal Panda_PandaGripper_libero_object; do
  $PY select_shards.py --shards $OUT/shards/$d --out $OUT/shards_$N/$d --n $N > logs/gs_select_$d.log 2>&1 || { echo "SELECT FAILED $d"; exit 1; }
  nice -n 19 $PY /workspace/yujin/FastWAM/libero_rot/pack_lerobot.py --shards $OUT/shards_$N/$d --out $OUT/lerobot_v30 --theta 0 \
    --name ${d}_no_noops_lerobot > logs/gs_pack_$d.log 2>&1 || { echo "PACK FAILED $d"; exit 1; }
done
echo ALL PACKED
