"""Initial-pose perturbation of LIBERO GT demos (LIBERO-Plus style).

A condition is a fixed Panda home ``HOME_Q + m * u`` (u: unit vector in joint
space, m in [0.1, 0.5] rad, the same distribution as LIBERO-Plus'
MountedPanda1..500). For every GT demo the robot starts from that home, and the
grip-site reference of the demo is warped so that it starts at the perturbed
grip pose and the offset decays to zero (smoothstep) by the merge step t_m, a few
steps before the first gripper close / first contact. After t_m the GT
trajectory is followed. The actions that are actually sent (GT action + warp
feed-forward + tracking feedback) are the new labels, so the approach segment has
different actions from the GT demo.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/workspace/yujin/libero-embodiment")

import mujoco
import numpy as np
from robosuite.utils.transform_utils import quat2axisangle

import replay
from replay import grip_frame, _rotvec

HOME_Q = np.array([0, -1.61037389e-01, 0.0, -2.44459747e00, 0.0, 2.22675220e00, np.pi / 4])
PRE_MERGE = 10  # merge this many steps before the first grasp / contact
MIN_MERGE = 15
MAX_FF = 0.3  # max per-step warp feed-forward in normalized action units
ROBOT_PREFIX = ("robot0_", "gripper0_", "mount0_")


def _expm(w):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, w / (np.linalg.norm(w) + 1e-12), np.linalg.norm(w))
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def condition_q0(mag, seed):
    """Fixed perturbed home for one condition: HOME_Q + mag * (seeded unit direction).

    ``mag`` is signed: -0.3 moves 0.3 rad along the opposite of the +0.3 direction.
    """
    u = np.random.default_rng(seed).normal(size=len(HOME_Q))
    return HOME_Q + mag * u / np.linalg.norm(u)


def robot_contacts(env):
    """Set of (robot geom, other geom) pairs currently in contact."""
    sim = env.env.sim
    pairs = set()
    for c in sim.data.contact[: sim.data.ncon]:
        n1, n2 = sim.model.geom_id2name(c.geom1) or "", sim.model.geom_id2name(c.geom2) or ""
        r1, r2 = n1.startswith(ROBOT_PREFIX), n2.startswith(ROBOT_PREFIX)
        if r1 != r2:
            pairs.add((n1, n2) if r1 else (n2, n1))
    return pairs


def gt_reference(bddl, demo, res=None):
    """Per-step GT grip-site pose, robot-contact sets and (optionally) upright GT frames."""
    env = replay.make_env(bddl, "panda_orig", demo["controller"], size=res or 64)
    env.reset()
    pos, rot, contacts, frames = [], [], [], dict(agentview=[], wrist=[])
    for st in demo["states"]:
        o = env.set_init_state(st)
        p, R, _, _ = grip_frame(env)
        pos.append(p)
        rot.append(R)
        contacts.append(robot_contacts(env))
        if res:
            frames["agentview"].append(o["agentview_image"][::-1].copy())
            frames["wrist"].append(o["robot0_eye_in_hand_image"][::-1].copy())
    env.close()
    return np.array(pos), np.array(rot), contacts, frames


def merge_step(demo, contacts):
    """A few steps before the first gripper close or first new robot contact."""
    close = np.where(demo["actions"][:, -1] > 0)[0]
    t_g = int(close[0]) if len(close) else len(demo["actions"])
    base = contacts[0]
    touch = [k for k, c in enumerate(contacts) if c - base]
    t_c = touch[0] if touch else len(demo["actions"])
    return max(min(t_g, t_c) - PRE_MERGE, MIN_MERGE), t_g, t_c


def set_arm_qpos(env, q):
    sim = env.env.sim
    r = env.env.robots[0]
    sim.data.qpos[r._ref_joint_pos_indexes] = q
    sim.data.qvel[r._ref_joint_vel_indexes] = 0
    sim.forward()
    # nullspace target = perturbed home, as with LIBERO-Plus' MountedPandaN init_qpos
    r.controller.update(force=True)
    r.controller.update_initial_joints(q)


def gt_episode(suite, task_id, demo_key, res=512):
    """Unperturbed (default) condition: the GT demo as is, re-rendered from its states.

    Same return layout as ``generate``; state/images at step k are taken at
    ``states[k]``, i.e. before ``actions[k]``.
    """
    t, bddl, h5 = replay.task_info(suite, task_id)
    demo = replay.load_demo(h5, demo_key)
    env = replay.make_env(bddl, "panda_orig", demo["controller"], size=res)
    env.reset()
    table = dict(ee_state=[], gripper_state=[], joint_state=[], action=demo["actions"])
    frames = dict(agentview=[], wrist=[])
    for st in demo["states"]:
        obs = env.set_init_state(st)
        table["ee_state"].append(np.hstack([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"])]))
        table["gripper_state"].append(obs["robot0_gripper_qpos"].copy())
        table["joint_state"].append(obs["robot0_joint_pos"].copy())
        frames["agentview"].append(obs["agentview_image"][::-1].copy())
        frames["wrist"].append(obs["robot0_eye_in_hand_image"][::-1].copy())
    env.set_init_state(demo["states"][-1])
    success = bool(env.check_success())
    env.close()
    table = {k: np.stack(v).astype(np.float32) for k, v in table.items()}
    info = dict(suite=suite, task_id=task_id, task=t.name, language=t.language, demo=demo_key,
                dq_norm=0.0, num_steps=int(len(demo["actions"])), success=success, valid=success)
    return info, table, frames, None


def generate(suite, task_id, demo_key, q0, res=512, kp=1.0, render_gt=False):
    """Roll out one demo from home ``q0``.

    Returns (info, table, frames, gt_frames). ``table`` holds per-step
    observation.state / states.* / action recorded *before* each action, and
    ``frames`` the upright ([::-1]) agentview / wrist images at the same steps.
    """
    t, bddl, h5 = replay.task_info(suite, task_id)
    demo = replay.load_demo(h5, demo_key)
    tpos, trot, contacts, gt_frames = gt_reference(bddl, demo, res if render_gt else None)
    t_m, t_g, t_c = merge_step(demo, contacts)

    env = replay.make_env(bddl, "panda_orig", demo["controller"], size=res)
    env.reset()
    env.set_init_state(demo["states"][0])
    set_arm_qpos(env, q0)
    start_contacts = robot_contacts(env) - contacts[0]
    obs = env.env._get_observations(force_update=True)

    pos_scale = demo["controller"]["output_max"][0]
    rot_scale = demo["controller"]["output_max"][3]
    p0, R0, _, _ = grip_frame(env)
    dp = p0 - tpos[0]
    w0 = _rotvec(R0 @ trot[0].T)

    # stretch the approach (hold the GT start) if the warp would need too large actions
    need = 1.5 * max(np.linalg.norm(dp) / pos_scale, np.linalg.norm(w0) / rot_scale) / MAX_FF
    n_pre = int(max(0, np.ceil(need - t_m)))
    hold = np.zeros((n_pre, 7))
    hold[:, -1] = demo["actions"][0, -1]
    acts = np.concatenate([hold, demo["actions"]])
    tpos = np.concatenate([np.repeat(tpos[:1], n_pre, 0), tpos])
    trot = np.concatenate([np.repeat(trot[:1], n_pre, 0), trot])
    T = t_m + n_pre
    s = _smoothstep(np.arange(len(acts) + 1) / T)

    table = dict(ee_state=[], gripper_state=[], joint_state=[], action=[])
    frames = dict(agentview=[], wrist=[])
    approach_contacts = set()
    track_err, success = [], False
    for k, a0 in enumerate(acts):
        # one step = record obs/state -> execute action
        table["ee_state"].append(np.hstack([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"])]))
        table["gripper_state"].append(obs["robot0_gripper_qpos"].copy())
        table["joint_state"].append(obs["robot0_joint_pos"].copy())
        frames["agentview"].append(obs["agentview_image"][::-1].copy())
        frames["wrist"].append(obs["robot0_eye_in_hand_image"][::-1].copy())

        p, R, _, _ = grip_frame(env)
        p_t = tpos[k] + (1 - s[k]) * dp
        R_t = _expm((1 - s[k]) * w0) @ trot[k]
        ds = s[k + 1] - s[k]
        a = a0.copy()
        a[:3] += (-ds * dp + kp * (p_t - p)) / pos_scale
        a[3:6] += (-ds * w0 + kp * _rotvec(R_t @ R.T)) / rot_scale
        a[:6] = np.clip(a[:6], -1, 1)
        track_err.append(np.linalg.norm(p_t - p))
        table["action"].append(a)
        obs, _, _, _ = env.step(a)
        if k < T:
            approach_contacts |= robot_contacts(env) - contacts[0]
        success = success or bool(env.check_success())
    env.close()

    table = {k: np.stack(v).astype(np.float32) for k, v in table.items()}
    gt_app = demo["actions"][:t_m]
    info = dict(
        suite=suite, task_id=task_id, task=t.name, language=t.language, demo=demo_key,
        dq_norm=float(np.linalg.norm(q0 - HOME_Q)), q0=q0.tolist(),
        start_eef_shift_cm=float(np.linalg.norm(dp) * 100), start_rot_deg=float(np.degrees(np.linalg.norm(w0))),
        t_merge=int(t_m), t_grasp=int(t_g), t_contact=int(t_c), n_pre_hold=int(n_pre),
        num_steps=int(len(acts)), gt_num_steps=int(len(demo["actions"])),
        approach_action_l2_vs_gt=float(np.linalg.norm(table["action"][n_pre:T, :6] - gt_app[:, :6], axis=1).mean()),
        after_merge_action_l2_vs_gt=float(np.linalg.norm(table["action"][T:, :6] - demo["actions"][t_m:, :6], axis=1).mean()),
        track_err_mm_after_merge_mean_max=[float(np.mean(track_err[T:]) * 1000), float(np.max(track_err[T:]) * 1000)],
        start_contacts=sorted({o for _, o in start_contacts}),
        approach_contacts=sorted({o for _, o in approach_contacts}),
        success=success,
    )
    info["valid"] = bool(success and not start_contacts and not approach_contacts)
    return info, table, frames, gt_frames
