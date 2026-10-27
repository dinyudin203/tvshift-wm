"""Sanity check for the OnRobot RG2 robosuite gripper (see README.md in this folder).

Builds LIBERO libero_goal task 1 with UR5e + OnRobotRG2Gripper (and Panda / UR5e + Robotiq 2F-85 for reference), then
prints: where the fingertips are relative to grip_site, the open / closed finger gap, the fingertip advance while closing,
and that the 4-bar mimic joints stay consistent.

  cd /workspace/yujin/embodiment_swap_v2 && MUJOCO_GL=egl python grippers/onrobot_rg2/check_rg2.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))  # embodiment_swap_v2/
from swap_core import *  # noqa: E402  (registers OnRobotRG2Gripper via common.py)


def tip_offset(env):
    """Finger colliders' extent along the grip-site approach axis (z), relative to grip_site [m]."""
    m, d = env.sim.model._model, env.sim.data._data
    s = nid(m, MJ.mjOBJ_SITE, 'gripper0_grip_site'); R, p = d.site_xmat[s].reshape(3, 3), d.site_xpos[s]
    lo, hi = np.inf, -np.inf
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, MJ.mjOBJ_GEOM, g) or ''
        b = mujoco.mj_id2name(m, MJ.mjOBJ_BODY, m.geom_bodyid[g]) or ''
        if not (m.geom_contype[g] and b.startswith('gripper0_') and ('finger' in b or 'pad' in n or 'follower' in b)):
            continue
        h = m.geom_size[g] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else m.geom_aabb[g][3:]
        off = m.geom_aabb[g][:3] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else np.zeros(3)
        c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * h + off
        z = (d.geom_xpos[g] + c @ d.geom_xmat[g].reshape(3, 3).T - p) @ R[:, 2]
        lo, hi = min(lo, z.min()), max(hi, z.max())
    return lo, hi


t, bddl = task_info('libero_goal', 1)
mk = lambda robot, grip: ControlEnv(bddl_file_name=bddl, robots=[robot], gripper_types=grip, use_camera_obs=False,
                                    has_offscreen_renderer=False, ignore_done=True, hard_reset=False)
envs = {}
for robot, grip in [('Panda', 'default'), ('UR5e', 'Robotiq85GripperMenagerieNoLinkCol'), ('UR5e', 'OnRobotRG2Gripper')]:
    e = mk(robot, grip); e.reset(); envs[grip] = e.env
    lo, hi = tip_offset(e.env)
    print(f'[tip] {robot}+{grip}: finger colliders {lo*1000:+.1f} .. {hi*1000:+.1f} mm along approach axis from grip_site')

src, dst = envs['default'], envs['OnRobotRG2Gripper']
sw = Swap(src, dst, 'UR5e')
print('[swap] panda_gripper', sw.panda_gripper, 'menagerie(linkage)', sw.menagerie, 'rg2', sw.rg2,
      'tip-advance table max %.1f mm' % (sw._tip_adv.max() * 1000))
type(dst.robots[0].gripper).speed = property(lambda self: 0.01)  # same slew as the data (--grip_speed 0.01)
m = dst.sim.model._model; m.opt.impratio = 10; m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC  # --menagerie_opt
J = lambda n: dst.sim.data._data.qpos[m.jnt_qposadr[nid(m, MJ.mjOBJ_JOINT, 'gripper0_' + n)]]
for name, g, n in [('open', -1, 20), ('close', 1, 60), ('reopen', -1, 60)]:
    for _ in range(n): dst.step(np.array([0, 0, 0, 0, 0, 0, g], dtype=np.float32))
    mimic = max(abs(J('right_driver_joint') - J('left_driver_joint')), abs(J('left_inner_knuckle_joint') - J('left_driver_joint')),
                abs(J('left_inner_finger_joint') + J('left_driver_joint')))
    print(f'[{name:6s}] driver {J("left_driver_joint"):+.3f} rad  gap {2*sw.panda_like_gripper_state()[0]*1000:5.1f} mm  '
          f'tip advance {sw.tip_advance()*1000:4.1f} mm  mimic err {mimic:.4f} rad')
