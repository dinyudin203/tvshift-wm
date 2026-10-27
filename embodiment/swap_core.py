"""Shared embodiment-swap helpers (same IK / object-copy / gripper mapping as replay_all.py), parameterised by env.

Used by the physics GT-replay check, the tvshift-format renderer and the eval env worker.
"""
from common import *

MJ = mujoco.mjtObj
def nid(m, t, n): return mujoco.mj_name2id(m, t, n)
def jnames(m): return [mujoco.mj_id2name(m, MJ.mjOBJ_JOINT, j) for j in range(m.njnt)]
def jdim(m, j): return {0: 7, 1: 4, 2: 1, 3: 1}[m.jnt_type[j]]
def rot_err(R, Rt): return 0.5 * sum(np.cross(R[:, i], Rt[:, i]) for i in range(3))


class PandaRef:
    """Reads Panda reference quantities (grip-site pose, finger gap, wrist cam) from a flattened Panda state."""
    def __init__(self, env):
        self.env = env; self.m, self.d = env.sim.model._model, env.sim.data._data
        self.site = nid(self.m, MJ.mjOBJ_SITE, 'gripper0_grip_site')
        self.cam = nid(self.m, MJ.mjOBJ_CAMERA, 'robot0_eye_in_hand')
        self.j1, self.j2 = (env.sim.model.get_joint_qpos_addr(n) for n in ('gripper0_finger_joint1', 'gripper0_finger_joint2'))
        # rigid offsets on the Panda hand, used to express any embodiment in Panda terms:
        # hand-body orientation w.r.t. the grip site (LIBERO ee_state uses the hand quat) and wrist cam w.r.t. the grip site
        mujoco.mj_forward(self.m, self.d); d = self.d; hand = nid(self.m, MJ.mjOBJ_BODY, 'robot0_right_hand')
        Rs, ps = d.site_xmat[self.site].reshape(3, 3), d.site_xpos[self.site]
        self.R_site_hand = Rs.T @ d.xmat[hand].reshape(3, 3)
        self.cam_rel = (Rs.T @ (d.cam_xpos[self.cam] - ps), Rs.T @ d.cam_xmat[self.cam].reshape(3, 3))
        self.fovy = self.m.cam_fovy[self.cam]

    def read(self, state):
        self.env.sim.set_state_from_flattened(state); self.env.sim.forward(); d = self.d
        J = np.zeros((6, self.m.nv)); mujoco.mj_jacSite(self.m, d, J[:3], J[3:], self.site)
        return dict(twist=J @ d.qvel,tp=d.site_xpos[self.site].copy(), tR=d.site_xmat[self.site].reshape(3, 3).copy(),
                    width=d.qpos[self.j1] - d.qpos[self.j2], fingers=d.qpos[[self.j1, self.j2]].copy(), qpos=d.qpos.copy(), arm=d.qpos[:7].copy(),
                    cp=d.cam_xpos[self.cam].copy(), cR=d.cam_xmat[self.cam].reshape(3, 3).copy())


class Swap:
    def __init__(self, src, dst, robot):
        self.robot_name = robot
        sm = src.sim.model._model; self.sm = sm
        self.dst = dst; dm, dd = dst.sim.model._model, dst.sim.data._data; self.dm, self.dd = dm, dd
        self.obj_j = [(nid(sm, MJ.mjOBJ_JOINT, n), nid(dm, MJ.mjOBJ_JOINT, n)) for n in jnames(dm)
                      if n and not n.startswith(('robot0_', 'gripper0_')) and nid(sm, MJ.mjOBJ_JOINT, n) >= 0]
        r = dst.robots[0]
        self.arm_q = np.array(r._ref_joint_pos_indexes); self.arm_v = np.array(r._ref_joint_vel_indexes)
        self.lo, self.hi = dm.jnt_range[[nid(dm, MJ.mjOBJ_JOINT, n) for n in r.robot_joints]].T
        self.q_ref = np.array(r.robot_model.init_qpos)
        self.site = nid(dm, MJ.mjOBJ_SITE, 'gripper0_grip_site')
        # Panda gripper (any arm): fingers copied 1:1 from the Panda state; Robotiq85: driver joints via the gap table
        self.panda_gripper = nid(dm, MJ.mjOBJ_JOINT, 'gripper0_finger_joint1') >= 0
        # upstream robosuite Robotiq85 (Robotiq85GripperOrig): rotating knuckles; its opening is left to the actuator
        self.orig_robotiq = nid(dm, MJ.mjOBJ_JOINT, 'gripper0_finger_joint') >= 0
        # Menagerie Robotiq 2F-85 (Robotiq85GripperMenagerie): hinge drivers in a 4-bar linkage, opening also left to the actuator
        jd = nid(dm, MJ.mjOBJ_JOINT, 'gripper0_right_driver_joint')
        self.menagerie = jd >= 0 and dm.jnt_type[jd] == mujoco.mjtJoint.mjJNT_HINGE
        # OnRobot RG2 (OnRobotRG2Gripper): same driver/pad names as the Menagerie 2F-85 (handled by the same code paths),
        # but its 4-bar is inner knuckle = driver, inner finger = -driver instead of spring_link / follower
        self.rg2 = nid(dm, MJ.mjOBJ_JOINT, 'gripper0_left_inner_finger_joint') >= 0
        self.linkage = self.orig_robotiq or self.menagerie
        names = ('gripper0_finger_joint1', 'gripper0_finger_joint2') if self.panda_gripper else \
                ('gripper0_finger_joint', 'gripper0_right_outer_knuckle_joint') if self.orig_robotiq else \
                ('gripper0_right_driver_joint', 'gripper0_left_driver_joint')
        self.gj = [nid(dm, MJ.mjOBJ_JOINT, n) for n in names]
        assert min(self.gj) >= 0
        self.cam = nid(dm, MJ.mjOBJ_CAMERA, 'robot0_eye_in_hand')
        self.cam_fovy0 = dm.cam_fovy[self.cam]
        if not self.panda_gripper and not self.linkage:
            self._gap_table()
        # 4-bar (Menagerie): the fingertips move along the approach axis as the fingers close; reference = open (reset)
        self._tip_pads = [nid(dm, MJ.mjOBJ_GEOM, 'gripper0_%s_pad1' % s_) for s_ in ('right', 'left')] if self.menagerie else []
        if self._tip_pads:
            self._tip_table()

    # ------------------------------------------------------------ gripper
    def set_drivers(self, s):
        for j in self.gj: self.dd.qpos[self.dm.jnt_qposadr[j]] = s

    def driver_qpos(self):
        return np.array([self.dd.qpos[self.dm.jnt_qposadr[j]] for j in self.gj])

    def _gap_table(self):
        dm, dd = self.dm, self.dd
        pa, pb = nid(dm, MJ.mjOBJ_GEOM, 'gripper0_right_pad1'), nid(dm, MJ.mjOBJ_GEOM, 'gripper0_left_pad1')
        saved = dd.qpos.copy(); self.thetas = np.linspace(0, 0.042, 43); g = []
        for th in self.thetas:
            self.set_drivers(th); mujoco.mj_forward(dm, dd); g.append(np.linalg.norm(dd.geom_xpos[pa] - dd.geom_xpos[pb]))
        dd.qpos[:] = saved; mujoco.mj_forward(dm, dd)
        self.gaps = np.array(g) - g[-1]

    def _pad_z(self, d):
        R = d.site_xmat[self.site].reshape(3, 3); p = d.site_xpos[self.site]
        return float(np.mean([(R.T @ (d.geom_xpos[g] - p))[2] for g in self._tip_pads]))

    def _tip_table(self):
        """Fingertip advance vs driver angle for the free (unloaded) 4-bar, where the fingers stay parallel:
        spring_link = driver, follower = -driver (checked against a physics close in free air)."""
        dm = self.dm; d = mujoco.MjData(dm); d.qpos[:] = self.dd.qpos
        adr = lambda s_, n: dm.jnt_qposadr[nid(dm, MJ.mjOBJ_JOINT, 'gripper0_%s_%s_joint' % (s_, n))]
        same, neg = ('inner_knuckle', 'inner_finger') if self.rg2 else ('spring_link', 'follower')
        # RG2 opens to driver -0.1 (reset/open state); the reference is the open position either way
        self._tip_th = np.linspace(-0.1, 0.8, 91) if self.rg2 else np.linspace(0, 0.8, 81); z = []
        for th in self._tip_th:
            for s_ in ('right', 'left'):
                d.qpos[adr(s_, 'driver')] = th; d.qpos[adr(s_, same)] = th; d.qpos[adr(s_, neg)] = -th
            mujoco.mj_kinematics(dm, d); z.append(self._pad_z(d))
        self._tip_adv = np.array(z) - z[0]
        self._drv = [dm.jnt_qposadr[nid(dm, MJ.mjOBJ_JOINT, 'gripper0_%s_driver_joint' % s_)] for s_ in ('right', 'left')]

    def tip_advance(self):
        """How far (m, along the grip-site approach axis) the fingertips of the unloaded linkage would be ahead of
        their open position at the current driver angle (i.e. how far the closing motion carries them). Uses the
        driver angle, not the pads, so a fingertip pushed back by an object does not change it. 0 for grippers whose
        pads only translate sideways (Panda, slide Robotiq)."""
        if not self._tip_pads: return 0.0
        return float(np.interp(np.mean(self.dd.qpos[self._drv]), self._tip_th, self._tip_adv))

    def theta_for(self, width): return float(np.interp(width, self.gaps[::-1], self.thetas[::-1]))
    def width_for(self, theta): return float(np.interp(theta, self.thetas, self.gaps))

    def panda_like_gripper_state(self):
        """Panda finger qpos; a Robotiq opening is expressed in Panda finger coordinates (+gap/2, -gap/2)."""
        if self.panda_gripper:
            return self.driver_qpos()
        if self.linkage:  # pad distance
            dm, dd = self.dm, self.dd
            pads = ('gripper0_left_fingerpad_collision', 'gripper0_right_fingerpad_collision') if self.orig_robotiq else \
                   ('gripper0_left_pad1', 'gripper0_right_pad1')
            w = np.linalg.norm(dd.geom_xpos[nid(dm, MJ.mjOBJ_GEOM, pads[0])] - dd.geom_xpos[nid(dm, MJ.mjOBJ_GEOM, pads[1])])
            if self.menagerie:  # pad-centre distance -> gap between the pad surfaces (box half-thickness 4 mm), as Panda's 2*q
                w -= 2 * dm.geom_size[nid(dm, MJ.mjOBJ_GEOM, pads[0])][1]
            return np.array([w / 2, -w / 2])
        w = self.width_for(self.driver_qpos().mean())
        return np.array([w / 2, -w / 2])

    # ------------------------------------------------------------ objects
    def copy_objs(self, qpos_src, zero_vel=False):
        sm, dm, dd = self.sm, self.dm, self.dd
        for sj, dj in self.obj_j:
            sa, da, k = sm.jnt_qposadr[sj], dm.jnt_qposadr[dj], jdim(dm, dj)
            dd.qpos[da:da + k] = qpos_src[sa:sa + k]
            if zero_vel:
                kv = {0: 6, 1: 3, 2: 1, 3: 1}[dm.jnt_type[dj]]; dd.qvel[dm.jnt_dofadr[dj]:dm.jnt_dofadr[dj] + kv] = 0

    # ------------------------------------------------------------ IK
    def _fk(self, q):
        self.dd.qpos[self.arm_q] = q; mujoco.mj_kinematics(self.dm, self.dd); mujoco.mj_comPos(self.dm, self.dd)
        return self.dd.site_xpos[self.site], self.dd.site_xmat[self.site].reshape(3, 3)

    def ik(self, q0, tp, tR, iters=100, lam=0.02):
        q = q0.copy(); J = np.zeros((6, self.dm.nv))
        for _ in range(iters):
            pos, R = self._fk(q)
            e = np.concatenate([tp - pos, rot_err(R, tR)])
            if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 2e-3: break
            mujoco.mj_jacSite(self.dm, self.dd, J[:3], J[3:], self.site)
            Ja = J[:, self.arm_v]
            dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + lam**2 * np.eye(6), e)
            if len(q) > 6:
                N = np.eye(len(q)) - np.linalg.pinv(Ja) @ Ja
                dq += 0.1 * N @ (self.q_ref - q)
            q = np.clip(q + np.clip(dq, -0.1, 0.1), self.lo, self.hi)
        return q, np.linalg.norm(e[:3]), np.linalg.norm(e[3:])

    def ik_priority(self, q, tp, tR):
        J = np.zeros((6, self.dm.nv))
        for w in (0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01):
            for _ in range(300):
                pos, R = self._fk(q)
                e = np.concatenate([tp - pos, w * rot_err(R, tR)])
                mujoco.mj_jacSite(self.dm, self.dd, J[:3], J[3:], self.site)
                Ja = J[:, self.arm_v].copy(); Ja[3:] *= w
                q = np.clip(q + np.clip(Ja.T @ np.linalg.solve(Ja @ Ja.T + 1e-3 * np.eye(6), e), -0.1, 0.1), self.lo, self.hi)
            pos, _ = self._fk(q)
            if np.linalg.norm(tp - pos) < 5e-4: break
        pos, R = self._fk(q)
        return q, np.linalg.norm(tp - pos), np.linalg.norm(rot_err(R, tR))

    def _ik_6d(self, q0, tp, tR):
        ok = lambda ep, er: ep < 5e-4 and er < 2e-3
        q, ep, er = self.ik(q0, tp, tR)
        if ok(ep, er): return q, ep, er
        best = (q, ep, er); rng = np.random.RandomState(0)
        for k in range(40):
            start = q0 if k < 3 else np.clip((self.q_ref if k % 2 else q0) + rng.normal(0, 0.4, len(q0)), self.lo, self.hi)
            q1, ep1, er1 = self.ik(start, tp, tR, iters=400, lam=0.05)
            if ok(ep1, er1) and np.abs(q1 - q0).max() < 0.5: return q1, ep1, er1
            if ep1 + 0.2 * er1 < best[1] + 0.2 * best[2]: best = (q1, ep1, er1)
        return best

    def ik_robust(self, q0, tp, tR):
        r = self._ik_6d(q0, tp, tR)
        if r[1] > 1e-3:
            return (*self.ik_priority(r[0], tp, tR), True)
        return (*r, False)

    def init_q(self, r):
        q = r['arm'].copy() if self.robot_name == 'Panda' else self.q_ref.copy()
        for _ in range(3): q, _, _ = self.ik(q, r['tp'], r['tR'], iters=300)
        return q

    # ------------------------------------------------------------ physics init from a Panda state
    def place_from_panda(self, r):
        """Physics env: objects from the Panda state, arm IK'd to the Panda grip pose, Robotiq opened like Panda."""
        dd, dm = self.dd, self.dm
        self.copy_objs(r['qpos'], zero_vel=True)
        q, ep, er, fb = self.ik_robust(self.init_q(r), r['tp'], r['tR'])
        dd.qpos[self.arm_q] = q; dd.qvel[self.arm_v] = 0
        # demo states start mid-motion: carry the Panda grip-site twist over to the new arm's joint velocities
        mujoco.mj_forward(dm, dd); J = np.zeros((6, dm.nv)); mujoco.mj_jacSite(dm, dd, J[:3], J[3:], self.site)
        dd.qvel[self.arm_v] = np.linalg.pinv(J[:, self.arm_v]) @ r['twist']
        if self.panda_gripper:
            for j, v in zip(self.gj, r['fingers']): dd.qpos[dm.jnt_qposadr[j]] = v
        elif self.linkage:
            pass
        else:
            self.set_drivers(self.theta_for(r['width']))
        dd.qvel[[dm.jnt_dofadr[j] for j in self.gj]] = 0
        mujoco.mj_forward(dm, dd)
        # the OSC caches the EEF pose from reset; refresh it (and the null-space home) or the first delta goal is
        # taken from the stale pose and the offset persists for the whole open-loop rollout
        c = self.dst.robots[0].controller
        c.update_initial_joints(q); c.update(force=True); c.reset_goal()
        return ep, er, fb

    def panda_equiv_state(self, ref):
        """8-d state in LIBERO/Panda convention: grip-site pos, axis-angle of the orientation the Panda hand would have,
        Panda-like finger qpos (+w/2, -w/2)."""
        import robosuite.utils.transform_utils as T
        dd = self.dd; R = dd.site_xmat[self.site].reshape(3, 3) @ ref.R_site_hand
        aa = T.quat2axisangle(T.mat2quat(R))
        return np.concatenate([dd.site_xpos[self.site], aa, self.panda_like_gripper_state()]).astype(np.float32)

    def joint_state7(self):
        q = self.dd.qpos[self.arm_q]; return np.pad(q, (0, 7 - len(q))).astype(np.float32)  # UR5e (6-DoF) zero-padded

    def align_wrist_cam(self, ref):
        """Put the wrist cam at the same pose relative to the grip site as on the Panda (rigid, so once per env)."""
        dm, dd = self.dm, self.dd; mujoco.mj_forward(dm, dd)
        Rs, ps = dd.site_xmat[self.site].reshape(3, 3), dd.site_xpos[self.site]
        self.move_wrist_cam(ps + Rs @ ref.cam_rel[0], Rs @ ref.cam_rel[1], ref.fovy); mujoco.mj_forward(dm, dd)

    def move_wrist_cam(self, cp, cR, fovy):
        dm, dd = self.dm, self.dd; body = dm.cam_bodyid[self.cam]
        dm.cam_fovy[self.cam] = fovy
        Rb, pb = dd.xmat[body].reshape(3, 3), dd.xpos[body]
        dm.cam_pos[self.cam] = Rb.T @ (cp - pb)
        qq = np.zeros(4); mujoco.mju_mat2Quat(qq, (Rb.T @ cR).ravel()); dm.cam_quat[self.cam] = qq
