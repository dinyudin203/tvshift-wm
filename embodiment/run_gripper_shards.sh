#!/bin/bash
# Arm + gripper embodiment data (IIWA+Robotiq 2F-85 spatial 2,6 / UR5e+OnRobot RG2 goal 1,7 / Panda+Panda object 7,8):
# all 50 demos per task, physics re-execution in the robotiq-run format; selection/packing is a separate step.
# CPU only (EGL = llvmpipe here), niced so the running ICWM training keeps priority.
cd /workspace/yujin/embodiment_swap_v2
export MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=/workspace/yujin/data/libero_embodiment_gripper
PY=/workspace/envs/fastwam/bin/python
RQ="--gripper Robotiq85GripperMenagerieNoLinkCol --menagerie_opt --tip_comp --grip_speed 0.01 --modes goal"
RG="--gripper OnRobotRG2Gripper --menagerie_opt --tip_comp --grip_speed 0.01 --modes goal"
for cfg in "libero_spatial 2 IIWA IIWA_Robotiq85" "libero_spatial 6 IIWA IIWA_Robotiq85" "libero_goal 1 UR5e UR5e_RG2" "libero_goal 7 UR5e UR5e_RG2" \
           "libero_object 7 Panda Panda_PandaGripper" "libero_object 8 Panda Panda_PandaGripper"; do
  set -- $cfg
  case $4 in IIWA_Robotiq85) F=$RQ;; UR5e_RG2) F=$RG;; *) F="";; esac
  for r in "0 25" "25 50"; do
    set -- $cfg $r
    nice -n 19 $PY render_emb_shard.py --suite $1 --task $2 --robot $3 --demo-start $5 --demo-end $6 $F \
      --out $OUT/shards/${4}_${1} > logs/gs_shard_${4}_${1}_t${2}_$5.log 2>&1 &
  done
done
wait
echo ALL SHARDS DONE
