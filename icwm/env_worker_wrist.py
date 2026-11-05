"""LIBERO environment server for Reflective VLA evaluation, + wrist-camera perturbation (tvshift v2wrist).

Copy of reflective_vla/eval/env_worker.py; a reset's "theta" may also be a string "a<front>_w<wrist>" (e.g. "a045_w030",
"a330_wm030"): agentview rotated by <front> deg as before, and the eye_in_hand camera rotated by <wrist> deg about the
gripper approach axis (hand-body z), exactly as tvshift/data/render_shard.py --wrist-deg renders the training data.

Runs in the *simulator* python (robosuite + LIBERO) and talks to the policy process over a
multiprocessing Connection, because the two stacks cannot share one environment. Observations
are produced exactly the way the training cache was: 512x512 render, vertical flip, INTER_AREA
resize to 224, state = [eef_pos, axis-angle(eef_quat), gripper_qpos] (8-d).

    python env_worker_wrist.py --port 5555 --libero-root /path/to/LIBERO --adapt-repo /path/to/lingbot-va-adapt
"""
import argparse
import os
import sys
from multiprocessing.connection import Client

import cv2
import numpy as np

AUTHKEY = b"reflective-eval"


def to_obs(raw, full=False):
    import robosuite.utils.transform_utils as T

    def img(x):
        return cv2.resize(np.ascontiguousarray(x[::-1]), (224, 224), interpolation=cv2.INTER_AREA)

    state = np.concatenate([raw["robot0_eef_pos"], T.quat2axisangle(raw["robot0_eef_quat"]),
                            raw["robot0_gripper_qpos"]]).astype(np.float32)
    out = {"image": img(raw["agentview_image"]), "wrist_image": img(raw["robot0_eye_in_hand_image"]),
           "state": state}
    if full:  # native-resolution frames (ICWM feeds 512x512 images to the Qwen processor)
        out["image_full"] = np.ascontiguousarray(raw["agentview_image"][::-1])
        out["wrist_image_full"] = np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1])
    return out


def parse_theta(theta):
    """-> (agentview azimuth deg, wrist deg)"""
    if isinstance(theta, str):
        a, w = theta.split("_")
        return float(a[1:]), (-1 if w[1] == "m" else 1) * float(w.lstrip("wm"))
    return float(theta), 0.0


def set_wrist_camera(model, p0, q0, deg):
    """absolute pose (from the unrotated p0 / q0), so repeated resets never accumulate"""
    import transforms3d.quaternions as quaternions

    wid = model.camera_name2id("robot0_eye_in_hand")
    a = np.deg2rad(deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    model.cam_pos[wid] = rz @ p0
    model.cam_quat[wid] = quaternions.qmult([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)], q0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--adapt-repo", default="/workspace/yujin/lingbot-va-adapt")
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()
    for p in (args.libero_root, args.adapt_repo):
        sys.path.insert(0, p)
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from online_adapt.env.conditions import apply_camera_condition

    conn = Client(("127.0.0.1", args.port), authkey=AUTHKEY)
    cache = {"key": None, "env": None, "task": None, "states": None, "full": False}

    def get_env(suite, idx):
        if cache["key"] != (suite, idx):
            if cache["env"] is not None:
                cache["env"].close()
            bm = benchmark.get_benchmark_dict()[suite]()
            task = bm.get_task(idx)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.res, camera_widths=args.res)
            env.seed(0)
            wid = env.env.sim.model.camera_name2id("robot0_eye_in_hand")  # unrotated wrist cam pose
            wrist0 = (env.env.sim.model.cam_pos[wid].copy(), env.env.sim.model.cam_quat[wid].copy())
            cache.update(key=(suite, idx), env=env, task=task, states=bm.get_task_init_states(idx), wrist0=wrist0)
        return cache["env"], cache["task"], cache["states"]

    while True:
        msg = conn.recv()
        cmd = msg["cmd"]
        if cmd == "close":
            if cache["env"] is not None:
                cache["env"].close()
            conn.send({"ok": True})
            break
        if cmd == "reset":
            cache["full"] = bool(msg.get("full", False))
            env, task, states = get_env(msg["suite"], msg["idx"])
            env.reset()
            # the rotation lives in sim.model, which LIBERO rebuilds on reset: apply after
            # every reset and before set_init_state (same as the WAM-IDM client)
            front, wrist = parse_theta(msg["theta"])
            if front != 0:
                apply_camera_condition(env, camera_name="agentview", azimuth_degrees=front)
            set_wrist_camera(env.env.sim.model, *cache["wrist0"], wrist)
            env.env.sim.forward()
            raw = env.set_init_state(states[msg["episode"] % len(states)])
            # Let objects settle. The WAM-IDM client waits with an all-zero action, but a zero gripper
            # command half-closes the fingers (qpos 0.021 vs 0.036 in the demos); wait with the
            # gripper commanded open so the state matches what training saw.
            for _ in range(5):
                raw, _, _, _ = env.step([0.0] * 6 + [-1.0])
            conn.send({"obs": to_obs(raw, cache["full"]), "prompt": task.language})
        elif cmd == "step":
            env = cache["env"]
            raw, reward, done, _ = env.step(list(map(float, msg["action"])))
            conn.send({"obs": to_obs(raw, cache["full"]), "done": bool(done), "t": int(env.env.timestep)})


if __name__ == "__main__":
    main()
