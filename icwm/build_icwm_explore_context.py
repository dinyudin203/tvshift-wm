"""Self-exploration context data for ICWM training on tvshift (context_source="explore").

The training-time counterpart of the eval's active probing: for every training viewpoint
(task, camera angle) of tvshift, reset LIBERO to a benchmark initial state, run the very same
`probe()` the eval runs (20 task-agnostic moves of 5 fixed-step delta-EEF actions toward random
targets, gripper open; run_icwm_eval_tvshift.py) through the same env_worker, and store the
recorded transitions. Initial states 10..49 are used, so the eval's states (episode e = state e,
e < 10) never appear in training context.

Output, one set per tvshift training dataset (= one viewpoint):
    <out>/<dataset name>__frames.npy   uint8 [S, 21, 512, 512, 3]  agentview o_0..o_20 (clip i = o_i, a_i, o_i+1)
    <out>/<dataset name>__actions.npy  float32 [S, 20, 5, 7]
    <out>/<dataset name>__meta.json    per session: prompt, init state, seed
and <out>/DONE when every viewpoint is written.

    python build_icwm_explore_context.py --out /dataset/yujin/data/tvshift_explore --gpus 0,1,2,3,4,5,6,7
"""
import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_icwm_eval_tvshift as ev  # noqa: E402  (probe(), EnvClient, TASKS: identical to the eval)

DATASET = {"libero_spatial": {0: "libero_spatial_theta000_no_noops_lerobot"},
           "libero_goal": {45: "libero_goal_theta045_no_noops_lerobot", 345: "libero_goal_theta345_no_noops_lerobot"},
           "libero_object": {270: "libero_object_theta270_no_noops_lerobot"}}

_LOCK = threading.Lock()


def build_task(args, gpu, task):
    tid, suite, idx, theta = task
    part = Path(args.out) / f"_part_t{tid}.npz"
    if part.exists():
        return part
    with _LOCK:  # EnvClient reads the render GPU from the environment
        os.environ["EVAL_RENDER_GPU"] = gpu
        envc = ev.EnvClient(args, gpu)
    frames, actions, meta = [], [], []
    for init in range(args.init_lo, args.init_hi):
        for s in range(args.sessions):
            seed = 1_000_000 * tid + 1000 * init + s
            rng = np.random.default_rng(seed)
            r = envc.call(cmd="reset", suite=suite, idx=idx, theta=theta, episode=init, full=True)
            clips = ev.probe(envc, r["obs"], rng)
            frames.append(np.stack([c[0] for c in clips] + [clips[-1][2]]))
            actions.append(np.stack([c[1] for c in clips]))
            meta.append({"task_id": tid, "prompt": r["prompt"], "init_state": init, "session": s, "seed": seed})
        print(f"[t{tid}] init {init} done", flush=True)
    envc.close()
    np.savez(part, frames=np.stack(frames), actions=np.stack(actions).astype(np.float32), meta=json.dumps(meta))
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--init-lo", type=int, default=10)
    ap.add_argument("--init-hi", type=int, default=50)
    ap.add_argument("--sessions", type=int, default=2, help="probing sessions per initial state")
    ap.add_argument("--env-python", default="/dataset/yujin/envs/tv/bin/python")
    ap.add_argument("--libero-root", default="/home/gpuuser/yujin/repos/LIBERO")
    ap.add_argument("--adapt-repo", default="/home/gpuuser/yujin/repos/lingbot-va-adapt")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus.split(",")
    with ThreadPoolExecutor(len(ev.TASKS)) as ex:
        parts = list(ex.map(lambda it: build_task(args, gpus[it[0] % len(gpus)], it[1]), enumerate(ev.TASKS)))
    by_ds = {}
    for task, part in zip(ev.TASKS, parts):
        by_ds.setdefault(DATASET[task[1]][task[3]], []).append(part)
    for name, ps in by_ds.items():
        zs = [np.load(p) for p in ps]
        fr = np.concatenate([z["frames"] for z in zs])
        np.save(out / f"{name}__frames.npy", fr)
        np.save(out / f"{name}__actions.npy", np.concatenate([z["actions"] for z in zs]))
        meta = [m for z in zs for m in json.loads(str(z["meta"]))]
        (out / f"{name}__meta.json").write_text(json.dumps(meta))
        print(f"{name}: {fr.shape[0]} sessions, frames {fr.shape}", flush=True)
    (out / "DONE").write_text("ok")


if __name__ == "__main__":
    main()
