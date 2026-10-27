# Arm + gripper embodiment-shift data, with an agentview angle variant (LIBERO)

Re-executes the Panda GT demos of LIBERO on another arm with another gripper (physics replay, the action labels are the
actions the new robot actually executed), and optionally rotates the agentview camera.

| dataset | arm + gripper | suite, task idx | camera (angle variant) |
|---|---|---|---|
| `IIWA_Robotiq85_libero_spatial` | IIWA + Robotiq 2F-85 (Menagerie 4-bar) | spatial 2, 6 | 0 deg |
| `UR5e_RG2_libero_goal` | UR5e + OnRobot RG2 | goal 1, 7 | 45 deg |
| `Panda_PandaGripper_libero_object` | Panda + Panda gripper | object 7, 8 | 270 deg |

First N = 45 successful demos per task (N = smallest per-task success count), 90 episodes per dataset.
Format and features: see `DATA_FORMAT.md` (same LeRobot v3.0 schema as tvshift; state in the Panda convention).

- `swap_core.py`, `common.py` - embodiment swap: object copy from the Panda init state, arm IK to the Panda grip pose,
  wrist cam at the Panda wrist-cam pose relative to the grip site, Panda-convention state (`Swap.panda_equiv_state`).
- `robotiq_orig.py`, `rg2_gripper.py`, `grippers/` - robosuite gripper models registered on import
  (`Robotiq85GripperMenagerieNoLinkCol`, `OnRobotRG2Gripper`, ...) and their MJCF / meshes (licenses in each folder).
- `render_emb_shard.py` - stage A: replay one task's demos on the swapped embodiment, per-step action retargeting to
  the Panda's absolute OSC goals, write shards (successes only, no-ops dropped). `--theta` rotates the agentview about
  the robot base z axis (libero_rot convention); `--menagerie_opt --tip_comp --grip_speed 0.01` are the 2F-85 / RG2 settings.
- `run_gripper_shards.sh` -> `chain_pack.sh` -> `pack_gripper.sh N` - the gripper data (default camera).
- `run_angle_shards.sh` - the angle variant (IIWA 0 deg shards reused, UR5e 45 deg, Panda 270 deg), select + pack.
- `select_shards.py` - keep the first N successful demos per task for packing.

External dependencies (not in this repo): `render_shard.py` / `pack_lerobot.py` (= `tvshift/data/`, FastWAM
`libero_rot`), LIBERO + robosuite 1.4 + MuJoCo, the raw LIBERO hdf5 demos; paths are hard-coded to the server layout.
