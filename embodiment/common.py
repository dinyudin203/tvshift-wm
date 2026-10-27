import os, sys, numpy as np, mujoco
sys.path.insert(0, '/workspace/yujin/hyunwoo/LIBERO')
os.environ.setdefault('MUJOCO_GL', 'egl')
import robosuite.models.robots as rm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

# IIWA / UR5e lack the LIBERO-specific table keys; reuse Panda's base placement so all arms sit at the Panda base.
from libero.libero.envs.robots.mounted_panda import MountedPanda
_pb = MountedPanda().base_xpos_offset
from robosuite.models.robots.manipulators.iiwa_robot import IIWA
from robosuite.models.robots.manipulators.ur5e_robot import UR5e
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm

# LIBERO prepends "Mounted" to the robot name; register Mounted variants that reuse Panda's base placement.
class MountedIIWA(IIWA):
    base_xpos_offset = property(lambda self: _pb)
class MountedUR5e(UR5e):
    base_xpos_offset = property(lambda self: _pb)
ROBOT_CLASS_MAPPING.update({'MountedIIWA': SingleArm, 'MountedUR5e': SingleArm})
# floor scenes (libero_object, ...) prepend "OnTheGround" instead: same base placement as OnTheGroundPanda, no mount
from libero.libero.envs.robots.on_the_ground_panda import OnTheGroundPanda
_pg = OnTheGroundPanda().base_xpos_offset
class OnTheGroundIIWA(IIWA):
    base_xpos_offset = property(lambda self: _pg)
    default_mount = property(lambda self: None)
class OnTheGroundUR5e(UR5e):
    base_xpos_offset = property(lambda self: _pg)
    default_mount = property(lambda self: None)
ROBOT_CLASS_MAPPING.update({'OnTheGroundIIWA': SingleArm, 'OnTheGroundUR5e': SingleArm})
import robotiq_orig  # noqa: F401  registers the upstream Robotiq85 as "Robotiq85GripperOrig"
import rg2_gripper  # noqa: F401  registers "OnRobotRG2Gripper"

def task_info(suite, tid):
    t = benchmark.get_benchmark_dict()[suite]().tasks[tid]
    bddl = os.path.join(get_libero_path('bddl_files'), t.problem_folder, t.bddl_file)
    return t, bddl

def make_env(bddl, robot, size, gripper=None):
    kw = dict(robots=[robot], camera_heights=size, camera_widths=size,
              camera_names=['agentview', 'robot0_eye_in_hand'])
    if gripper:
        kw['gripper_types'] = gripper
    env = ControlEnv(bddl_file_name=bddl, **kw)
    env.reset()
    return env.env
