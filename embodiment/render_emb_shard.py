"""Stage A (embodiment version of FastWAM/libero_rot/render_shard.py): re-execute the Panda GT trajectory on a swapped
embodiment (arm + --gripper, default the Panda gripper) with physics and write tvshift-format shards (512px h264, no-op filtered, successes only).

Actions are re-targeted per step so the new robot aims at the Panda's absolute OSC goals (see rollout()); --modes
adds fallbacks for demos that fail that way.
Init = objects from the demo's first Panda state, arm IK'd to the Panda grip pose with the Panda grip-site twist.
Obs are recorded before each action (same as the eval env): agentview at the default camera, wrist cam at the Panda
wrist-cam pose relative to the grip site, state in Panda convention (Swap.panda_equiv_state).
Output layout matches pack_lerobot.py (--theta 0): <out>/{theta000,wrist,frames,_done}/task{TT}_demo{DDD}.*

  MUJOCO_GL=egl python render_emb_shard.py --suite libero_goal --task 4 --robot UR5e --out .../shards/UR5e_goal
"""
import argparse, glob, json, sys, time, h5py
from swap_core import *
sys.path.insert(0, '/workspace/yujin/FastWAM/libero_rot')
from render_shard import is_noop, write_mp4, rotated_camera_pose
import robosuite.utils.transform_utils as T

p = argparse.ArgumentParser()
p.add_argument('--suite'); p.add_argument('--task', type=int); p.add_argument('--robot')
p.add_argument('--demo-start', type=int, default=0); p.add_argument('--demo-end', type=int, default=10**9)
p.add_argument('--res', type=int, default=512); p.add_argument('--fps', type=int, default=20)
p.add_argument('--out', required=True)
p.add_argument('--theta', type=int, default=0, help='agentview camera angle shift (deg about the robot base z axis, FastWAM/libero_rot convention); 0 = default camera')
p.add_argument('--gripper', default='PandaGripper', help='PandaGripper (same hand on every arm) or Robotiq85Gripper')
p.add_argument('--menagerie_opt', action='store_true', help='impratio=10, elliptic cone (the Menagerie 2F-85 scene option, dropped by robosuite)')
p.add_argument('--tip_comp', action='store_true', help='grip-site target shifted back by the 4-bar fingertip advance (Swap.tip_advance)')
p.add_argument('--grip_speed', type=float, default=0, help='gripper command slew per sim substep (Robotiq default 0.2, Panda 0.01)')
p.add_argument('--modes', default='goal,goal_fb,open', help="tried in order until the demo succeeds, e.g. goal,goal_fb,open")
a = p.parse_args()

task, bddl = task_info(a.suite, a.task)
raw = glob.glob(f'/workspace/yujin/data/libero_raw/{a.suite}/{task.name}_demo.hdf5')[0]
f = h5py.File(raw, 'r')
demos = sorted(f['data'].keys(), key=lambda k: int(k.split('_')[1]))[a.demo_start:a.demo_end]

def phys_env(robot, gripper, render):
    e = ControlEnv(bddl_file_name=bddl, robots=[robot], gripper_types=gripper, use_camera_obs=False,
                   has_offscreen_renderer=render, ignore_done=True, horizon=2000, hard_reset=False)
    e.reset(); return e.env

src = phys_env('Panda', 'default', False); ref = PandaRef(src)
dst = phys_env(a.robot, a.gripper, True); sw = Swap(src, dst, a.robot)
sw.align_wrist_cam(ref)
if a.theta:
    # same camera pose on every arm: computed once from the Panda env (robot0_link0 pose is identity there, as in
    # libero_rot/render_shard.py and the eval env's _rotate_agentview_sim; IIWA / UR5e have no robot0_link0 body)
    sm, dm_ = src.sim.model, dst.sim.model
    cid, did, bid = sm.camera_name2id('agentview'), dst.sim.model.camera_name2id('agentview'), sm.body_name2id('robot0_link0')
    assert np.allclose(sm.cam_pos[cid], dm_.cam_pos[did]) and np.allclose(sm.cam_quat[cid], dm_.cam_quat[did]), 'agentview differs between envs'
    pos, quat = rotated_camera_pose(sm.cam_pos[cid].copy(), sm.cam_quat[cid].copy(), sm.body_pos[bid].copy(), sm.body_quat[bid].copy(), a.theta)
    dm_.cam_pos[did] = pos; dm_.cam_quat[did] = quat
if a.menagerie_opt:
    dst.sim.model._model.opt.impratio = 10; dst.sim.model._model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
if a.grip_speed:
    type(dst.robots[0].gripper).speed = property(lambda self, v=a.grip_speed: v)
dd = sw.dd; osc = dst.robots[0].controller; osc.scale_action(np.zeros(6))  # initialises action_scale
render = lambda cam: dst.sim.render(camera_name=cam, width=a.res, height=a.res)[::-1].copy()

t0 = time.time()
for dn in demos:
    di = int(dn.split('_')[1]); stem = f'task{a.task:02d}_demo{di:03d}'
    done = os.path.join(a.out, '_done', f'{stem}.json')
    if os.path.exists(done): print('[skip]', stem, flush=True); continue
    states, actions = f['data'][dn]['states'][()], f['data'][dn]['actions'][()].astype(np.float32)
    keep, prev = [], None
    for t in range(len(actions)):
        if not is_noop(actions[t], prev): keep.append(t)
        prev = actions[t]
    keep = set(keep)

    def rollout(mode):
        """mode: 'goal' = aim at the Panda's absolute OSC goals; 'goal_fb' = same goals shifted by the current tracking
        error (pushes harder while lagging); 'open' = the Panda actions as recorded. The label is always what ran."""
        dst.reset()
        ep, er, fb = sw.place_from_panda(ref.read(states[0]))
        if not sw.panda_gripper:  # Panda gripper: keep the reset value, as when the demos were recorded
            dst.robots[0].gripper.current_action = np.array([-1.0])
        L = dict(ag=[], wr=[], st=[], js=[], act=[]); succ_t = -1; G = None; track_err = []
        for t, a_p in enumerate(actions):
            # goal matching: the Panda OSC at step t aimed at an absolute goal (pos = grip pos + scaled delta; ori =
            # rotated grip ori, kept from the previous step when the rotation delta is exactly 0). Re-express that same
            # absolute goal as a delta from the new robot's current grip pose, so tracking errors do not accumulate.
            r = ref.read(states[t]); sd = osc.scale_action(a_p[:6])
            gp = r['tp'] + sd[:3]
            if G is None or np.any(sd[3:] != 0): G = T.quat2mat(T.axisangle2quat(sd[3:])) @ r['tR']
            if a.tip_comp: gp = gp - G[:, 2] * sw.tip_advance()  # fingertips, not the grip site, follow the Panda's
            p_new, R_new = dd.site_xpos[sw.site].copy(), dd.site_xmat[sw.site].reshape(3, 3).copy()
            track_err.append(np.linalg.norm(p_new - r['tp']))
            if mode == 'open':
                u = a_p.copy()
            else:
                gp_, G_ = gp, G
                if mode == 'goal_fb':
                    gp_ = gp + (r['tp'] - p_new); G_ = (r['tR'] @ R_new.T) @ G
                u = np.concatenate([(gp_ - p_new) / osc.action_scale[:3],
                                    T.quat2axisangle(T.mat2quat(G_ @ R_new.T)) / osc.action_scale[3:], a_p[6:]])
            u = np.clip(u, -1, 1).astype(np.float32)
            if t in keep:
                L['st'].append(sw.panda_equiv_state(ref)); L['js'].append(sw.joint_state7()); L['act'].append(u)
                L['ag'].append(render('agentview')); L['wr'].append(render('robot0_eye_in_hand'))
            dst.step(u)
            if succ_t < 0 and dst._check_success(): succ_t = t
        return succ_t, L, track_err, ep, fb

    for mode in a.modes.split(','):
        succ_t, L, track_err, ep, fb = rollout(mode)
        if succ_t >= 0: break
        print(f'[retry] {stem} mode={mode} failed track={np.mean(track_err)*1000:.1f}mm', flush=True)
    ag, wr, act, js = L['ag'], L['wr'], L['act'], L['js']
    st = L['st']; ee = [x[:6] for x in st]; gs = [x[6:] for x in st]
    if succ_t < 0: print(f'[drop-fail] {stem} track={np.mean(track_err)*1000:.1f}/{np.max(track_err)*1000:.1f}mm', flush=True); continue
    if len(act) < 8: print(f'[drop-short] {stem}', flush=True); continue

    write_mp4(os.path.join(a.out, f'theta{a.theta:03d}', f'{stem}.mp4'), ag, a.fps)
    write_mp4(os.path.join(a.out, 'wrist', f'{stem}.mp4'), wr, a.fps)
    os.makedirs(os.path.join(a.out, 'frames'), exist_ok=True)
    np.savez(os.path.join(a.out, 'frames', f'{stem}.npz'), **{
        'observation.state': np.array(st), 'observation.states.ee_state': np.array(ee),
        'observation.states.joint_state': np.array(js), 'observation.states.gripper_state': np.array(gs),
        'action': np.array(act)})
    os.makedirs(os.path.dirname(done), exist_ok=True)
    json.dump(dict(task=a.task, demo=di, stem=stem, language=task.language, length=len(act), raw_length=len(actions),
                   thetas=[a.theta], res=a.res, fps=a.fps, robot=a.robot, gripper=a.gripper, suite=a.suite,
                   success_step=succ_t, action_mode=mode, menagerie_opt=a.menagerie_opt, tip_comp=a.tip_comp, grip_speed=a.grip_speed or None, track_err_mean_m=float(np.mean(track_err)), track_err_max_m=float(np.max(track_err)), init_ik_pos_err_m=float(ep), init_ik_fallback=bool(fb)), open(done, 'w'))
    print(f'[ok] {stem} kept={len(act)}/{len(actions)} succ@{succ_t} track={np.mean(track_err)*1000:.1f}/{np.max(track_err)*1000:.1f}mm elapsed={time.time()-t0:.0f}s', flush=True)
print('shard done', flush=True)
