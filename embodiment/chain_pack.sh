#!/bin/bash
# After run_gripper_shards.sh: N = smallest per-task success count, then pack_gripper.sh N.
cd /workspace/yujin/embodiment_swap_v2
until grep -q "ALL SHARDS DONE" logs/gs_driver.log 2>/dev/null; do sleep 30; done
S=/workspace/yujin/data/libero_embodiment_gripper/shards
N=$(for d in IIWA_Robotiq85_libero_spatial:2 IIWA_Robotiq85_libero_spatial:6 UR5e_RG2_libero_goal:1 UR5e_RG2_libero_goal:7 Panda_PandaGripper_libero_object:7 Panda_PandaGripper_libero_object:8; do
  ls $S/${d%%:*}/_done/task$(printf %02d ${d##*:})_demo*.json | wc -l; done | sort -n | head -1)
echo "N=$N"
./pack_gripper.sh $N
