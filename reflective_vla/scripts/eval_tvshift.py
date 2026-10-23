"""Reflective VLA evaluation on the tvshift protocols (seen pair + novel pair).

  seen  : each of the 8 training tasks at the single angle it was trained at       (8 combos)
  novel : each task at the 3 other trained angles -- seen task, seen angle, unseen pairing (24)

Parent mode launches one process per GPU; each loads the policy once and works through its
share of combos, driving a LIBERO simulator subprocess (env_worker.py) over a local socket.
Finished combos are skipped, so re-running resumes.

    python scripts/eval_tvshift.py --run-dir outputs/tvshift_K8 --out evaluate_results/reflective_tvshift \
        --gpus 0,1,2,3 --episodes 10 --env-python /opt/conda/envs/lingbot-va/bin/python \
        --libero-root /workspace/lerobot-libero
Test the plumbing without a trained model: --init-only /workspace/yujin/checkpoints/pi05_base
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TASKS = [(0, "libero_spatial", 1, 0), (1, "libero_spatial", 3, 0),
         (2, "libero_goal", 0, 45), (3, "libero_goal", 7, 45),
         (4, "libero_goal", 5, 345), (5, "libero_goal", 2, 345),
         (6, "libero_object", 0, 270), (7, "libero_object", 1, 270)]
ANGLES = [0, 45, 270, 345]
HORIZON = 800  # env steps, same as the WAM-IDM client
AUTHKEY = b"reflective-eval"
# the tvshift training data was rendered with MuJoCo 3.3.2; MuJoCo 3.8.1 (lingbot-va env) renders the
# libero_object floor scene much darker (agentview mean 137.7 -> 102.5). ENV_WORKER_PYTHONPATH="" = old behaviour
MUJOCO332 = os.environ.get("ENV_WORKER_PYTHONPATH", "/workspace/yujin/envs/mujoco332")
# TVSHIFT_ONLY=tag1,tag2: run just these combos; REFL_SAVE_VIDEO=dir: save one mp4 per episode
ONLY = set(os.environ["TVSHIFT_ONLY"].split(",")) if os.environ.get("TVSHIFT_ONLY") else None
SAVE_VIDEO = os.environ.get("REFL_SAVE_VIDEO")
# TVSHIFT_SWEEP=1: angle sweep instead -- every task at the 11 untrained angles of the ICWM / WAM sweep
# (TVSHIFT_SWEEP_TASKS=6,7 restricts the tasks)
SWEEP_ANGLES = [30, 60, 90, 120, 135, 225, 255, 285, 300, 315, 330]
SWEEP_TASKS = [int(x) for x in os.environ["TVSHIFT_SWEEP_TASKS"].split(",")] if os.environ.get("TVSHIFT_SWEEP_TASKS") \
    else [t for t, *_ in TASKS]


def combos():
    if os.environ.get("TVSHIFT_SWEEP"):
        out = [("sweep", t, s, i, th) for t, s, i, own in TASKS if t in SWEEP_TASKS for th in SWEEP_ANGLES]
        return [c for c in out if tag_of(c) in ONLY] if ONLY is not None else out
    out = [("seen", t, s, i, own) for t, s, i, own in TASKS]
    out += [("novel", t, s, i, th) for t, s, i, own in TASKS for th in ANGLES if th != own]
    return [c for c in out if tag_of(c) in ONLY] if ONLY is not None else out


def tag_of(c):
    return f"{c[0]}_t{c[1]}_{c[2]}_idx{c[3]}_theta{c[4]:03d}"


def summary_file(out, c):
    return Path(out) / tag_of(c) / f"{c[2]}_summary.json"


# ----------------------------------------------------------------------------- worker side
class EnvClient:
    def __init__(self, args, gpu):
        self.listener = Listener(("127.0.0.1", 0), authkey=AUTHKEY)
        port = self.listener.address[1]
        env = dict(os.environ, MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="0",  # single EGL device on hilo2
                   CUDA_VISIBLE_DEVICES="0", PYTHONPATH=MUJOCO332)
        self.proc = subprocess.Popen(
            [args.env_python, str(ROOT / "src/reflective_vla/eval/env_worker.py"), "--port", str(port),
             "--libero-root", args.libero_root, "--adapt-repo", args.adapt_repo], env=env)
        self.conn = self.listener.accept()

    def call(self, **msg):
        self.conn.send(msg)
        return self.conn.recv()

    def close(self):
        try:
            self.call(cmd="close")
        except Exception:
            pass
        self.proc.wait(timeout=60)


def load_policy(args):
    import torch
    from safetensors.torch import load_file

    from lerobot.configs.policies import PreTrainedConfig
    from reflective_vla import ReflectiveVLAConfig, ReflectiveVLAPolicy
    from reflective_vla.dtype_utils import add_input_casts

    run = Path(args.run_dir)
    cfg = PreTrainedConfig.from_pretrained(str(run / "config"))
    cfg.device = "cuda"
    cfg.pretrained_path = None
    if args.init_only:
        policy = ReflectiveVLAPolicy.from_pretrained(args.init_only, config=cfg, strict=False)
    else:
        exports = sorted((run / "export").glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
        d = Path(args.export) if args.export else exports[-1]
        policy = ReflectiveVLAPolicy(cfg)
        missing, unexpected = policy.load_state_dict(load_file(str(d / "model.safetensors")), strict=False)
        print(f"loaded {d}: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    policy.eval()
    add_input_casts(policy)
    stats = json.loads((run / "stats.json").read_text())
    stats = {k: {a: np.asarray(b, np.float32) for a, b in v.items()} for k, v in stats.items()}
    return policy, cfg, stats


def norm(x, s):
    return np.clip(2 * (x - s["q01"]) / np.maximum(s["q99"] - s["q01"], 1e-6) - 1, -1.5, 1.5).astype(np.float32)


def unnorm(a, s):
    return (a + 1) / 2 * (s["q99"] - s["q01"]) + s["q01"]


def run_episode(policy, cfg, stats, tok, envc, c, episode):
    import torch

    proto, tid, suite, idx, theta = c
    r = envc.call(cmd="reset", suite=suite, idx=idx, theta=theta, episode=episode)
    obs, prompt = r["obs"], r["prompt"]
    ids = tok(prompt, max_length=cfg.tokenizer_max_length, padding="max_length", truncation=True, return_tensors="pt")
    policy.reset()
    done, t, frames = False, 0, []
    while t < HORIZON and not done:
        batch = {
            "observation.images.image": torch.from_numpy(obs["image"]).permute(2, 0, 1)[None].float().div(255),
            "observation.images.wrist_image": torch.from_numpy(obs["wrist_image"]).permute(2, 0, 1)[None].float().div(255),
            "observation.state": torch.from_numpy(norm(obs["state"], stats["state"]))[None],
            "observation.language.tokens": ids["input_ids"],
            "observation.language.attention_mask": ids["attention_mask"].bool(),
        }
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            a = policy.select_action(batch)[0].float().cpu().numpy()  # normalised [7]
        a = unnorm(np.clip(a, -1.0, 1.0), stats["action"])
        a[6] = 1.0 if a[6] > 0 else -1.0  # LIBERO gripper is binary
        r = envc.call(cmd="step", action=a)
        obs, done, t = r["obs"], r["done"], r["t"]
        if SAVE_VIDEO:
            frames.append(np.concatenate([obs["image"], obs["wrist_image"]], 1))
    if SAVE_VIDEO:
        import imageio

        Path(SAVE_VIDEO).mkdir(parents=True, exist_ok=True)
        imageio.mimsave(f"{SAVE_VIDEO}/{tag_of(c)}_ep{episode}_{'success' if done else 'fail'}.mp4", frames,
                        fps=20, codec="libx264", macro_block_size=1)
    return bool(done), prompt


def worker_main(args):
    from transformers import AutoTokenizer

    gpu = args.gpu
    policy, cfg, stats = load_policy(args)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    envc = EnvClient(args, gpu)
    mine = combos()[args.bucket::args.nbuckets]
    t0 = time.time()
    for c in mine:
        sf = summary_file(args.out, c)
        if sf.exists():
            continue
        wins, prompt = 0, ""
        for ep in range(args.episodes):
            ok, prompt = run_episode(policy, cfg, stats, tok, envc, c, ep)
            wins += ok
            print(f"[{tag_of(c)}] ep {ep}: {'success' if ok else 'fail'} ({wins}/{ep + 1}) "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text(json.dumps({"benchmark": c[2], "tasks": [{
            "task_id": c[1], "prompt": prompt, "succ_num": float(wins), "total_num": float(args.episodes),
            "succ_rate": wins / args.episodes}]}))
    envc.close()


# ----------------------------------------------------------------------------- parent side
def parent_main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus.split(",")
    procs = []
    for i, g in enumerate(gpus):
        cmd = [sys.executable, __file__, "--worker", "--gpu", g, "--bucket", str(i), "--nbuckets", str(len(gpus))]
        for k in ("run_dir", "out", "episodes", "env_python", "libero_root", "adapt_repo", "tokenizer", "export", "init_only"):
            v = getattr(args, k)
            if v not in (None, ""):
                cmd += [f"--{k.replace('_', '-')}", str(v)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, TOKENIZERS_PARALLELISM="false")
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(out / f"worker{g}.log", "a"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    rows, tot = [], {"seen": [0, 0], "novel": [0, 0], "sweep": [0, 0]}
    for c in combos():
        sf = summary_file(out, c)
        s = json.loads(sf.read_text())["tasks"][0] if sf.exists() else None
        rows.append({"protocol": c[0], "task_id": c[1], "suite": c[2], "libero_idx": c[3], "theta": c[4],
                     "successes": s["succ_num"] if s else None, "episodes": s["total_num"] if s else None})
        if s:
            tot[c[0]][0] += s["succ_num"]
            tot[c[0]][1] += s["total_num"]
    missing = [tag_of(c) for c, r in zip(combos(), rows) if r["successes"] is None]
    if missing:  # a worker died: no summary.json, so a driver that polls for it relaunches and resumes
        print(f"INCOMPLETE: {len(missing)} combos missing, e.g. {missing[:3]}", flush=True)
        sys.exit(1)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    for p, (a, b) in tot.items():
        print(f"  {p:6s}: {a:.0f}/{b:.0f} = {100 * a / max(b, 1):.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--env-python", default="/opt/conda/envs/lingbot-va/bin/python")
    ap.add_argument("--libero-root", default="/workspace/lerobot-libero")
    ap.add_argument("--adapt-repo", default="/workspace/yujin/lingbot-va-adapt")
    ap.add_argument("--tokenizer", default="/workspace/yujin/checkpoints/paligemma_tokenizer")
    ap.add_argument("--export", default="", help="export/step_N dir (default: newest)")
    ap.add_argument("--init-only", default="", help="evaluate a pi0.5 init instead of a trained export (plumbing test)")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--bucket", type=int, default=0)
    ap.add_argument("--nbuckets", type=int, default=1)
    a = ap.parse_args()
    worker_main(a) if a.worker else parent_main(a)
