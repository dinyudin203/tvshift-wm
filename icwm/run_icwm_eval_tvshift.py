"""ICWM evaluation on the tvshift protocols (seen pair + novel pair), with active probing.

  seen  : each of the 8 training tasks at the single angle it was trained at       (8 combos)
  novel : each task at the 3 other trained angles -- seen task, seen angle, unseen pairing (24)

Per episode (paper Sec 4.2):
  1. active probing: in the *test* environment (the rotated camera), execute 20 task-agnostic
     probing moves -- 5 fixed-step delta-EEF actions toward a random target in the workspace --
     and record (o^s, a, o^e) for each. Probing may touch objects, so the episode is then reset
     to the identical initial state; only the recorded clips carry over (the paper requires the
     task's initial state to stay undisturbed).
  2. in-context execution: every time a new action chunk is needed, N=5 clips are drawn at
     random from the 20 and prepended to the query (o_t, l); the policy decodes FAST tokens.

History mode (--history-context, Reflective-VLA style context): no probing; the N clips are the N
chunks this episode has executed most recently, (o before the chunk, its 5 actions, o after), oldest
first; while fewer than N chunks have run, only those are used (none at the first chunk).

Mixed mode (--mixed-context, models trained with context_source=mixed): both -- N probing clips drawn
as in the default mode ("Exploration:" segment) followed by the N most recent executed chunks
("History:" segment). --mixed-keep explore|history drops the other segment (context-dropout ablation).

Parent mode launches one process per GPU; each loads the policy once and drives a LIBERO
simulator subprocess (reflective_vla/eval/env_worker.py) over a local socket. Finished combos
are skipped, so re-running resumes.

    python run_icwm_eval_tvshift.py --ckpt <.../checkpoints/last/pretrained_model> \
        --out evaluate_results/icwm_tvshift --gpus 0,1,2,3 --episodes 10
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

TASKS = [(0, "libero_spatial", 1, 0), (1, "libero_spatial", 3, 0),
         (2, "libero_goal", 0, 45), (3, "libero_goal", 7, 45),
         (4, "libero_goal", 5, 345), (5, "libero_goal", 2, 345),
         (6, "libero_object", 0, 270), (7, "libero_object", 1, 270)]
ANGLES = [0, 45, 270, 345]
HORIZON = 800  # env steps of task rollout (probing steps excluded), same as the other models
AUTHKEY = b"reflective-eval"
# the tvshift training data was rendered with MuJoCo 3.3.2: render the eval env with it too (see EnvClient)
MUJOCO332 = "/workspace/yujin/envs/mujoco332"
ENV_WORKER = "/workspace/yujin/reflectiveVLA/src/reflective_vla/eval/env_worker.py"

N_PROBES = 20          # probing moves recorded once per episode
PROBE_STEPS = 5        # = action_chunk_size, the clip horizon used in training
PROBE_SCALE = 0.5      # fixed step magnitude (delta-EEF action units)
N_CONTEXT = 5          # clips drawn per inference step
# probing targets: a box around the EE start pose of the episode. It was first fixed in world
# coordinates (x -0.35..0.05, y -0.25..0.25, z 1.00..1.30), which fits the table scenes (spatial/goal,
# EE starts at about (-0.21, 0.0, 1.18)) but not the floor scene of libero_object (EE starts at
# z ~0.25): every probe there pointed straight up and the arm left the camera view after ~5 probes.
# The box is now that same box expressed relative to PROBE_REF, re-anchored at each episode's EE start.
PROBE_LO = np.array([-0.35, -0.25, 1.00])
PROBE_HI = np.array([0.05, 0.25, 1.30])
PROBE_REF = np.array([-0.21, 0.0, 1.18])  # EE start pose of the table scenes the box was written for


# context-swap study: the task is always executed in its own env (suite, idx, theta); only the env the
# probing clips are recorded in changes. name -> (exec suite, idx, theta, probe suite, idx, theta)
CTX_SWAP = {
    "own_ctx":        ("libero_object", 0, 270, "libero_object", 0, 270),  # baseline: same scene, same camera
    "ctx_theta345":   ("libero_object", 0, 270, "libero_object", 0, 345),  # same scene, camera at 345 deg
    "ctx_obj1cheese": ("libero_object", 0, 270, "libero_object", 1, 270),  # other task's scene (cheese), same camera
}


def farthest_angle(theta):
    """The trained angle farthest (circularly) from theta: context recorded with the wrong camera."""
    return max((a for a in ANGLES if a != theta), key=lambda a: min(abs(a - theta), 360 - abs(a - theta)))


# angle sweep (TVSHIFT_SWEEP=1): one representative task per suite at angles outside the 4 trained ones
SWEEP_TASKS = [0, 2, 6]   # spatial idx1, goal idx0, object idx0
SWEEP_TASKS_ACTIVE = [int(x) for x in os.environ["TVSHIFT_SWEEP_TASKS"].split(",")] if os.environ.get("TVSHIFT_SWEEP_TASKS") else SWEEP_TASKS
SWEEP_ANGLES = [30, 60, 90, 120, 135, 225, 255, 285, 300, 315, 330]


# TVSHIFT_TASKS=6,7: restrict the seen/novel protocol to these task ids (e.g. a partial re-run)
TASKS_ACTIVE = [int(x) for x in os.environ["TVSHIFT_TASKS"].split(",")] if os.environ.get("TVSHIFT_TASKS") else [t[0] for t in TASKS]


# ICWM_SAVE_VIDEO=<dir>: write one mp4 per episode (agentview | wrist, context o^s strip underneath)
SAVE_VIDEO = os.environ.get("ICWM_SAVE_VIDEO")
# ablation: black out the agentview camera everywhere (policy input and probing clips) -> how much
# the policy relies on the third-person view vs wrist camera + proprio state
BLANK_AGENTVIEW = bool(os.environ.get("ICWM_BLANK_AGENTVIEW"))
# TVSHIFT_ONLY=tag1,tag2: run just these combos (tags as in tag_of)
ONLY = set(os.environ["TVSHIFT_ONLY"].split(",")) if os.environ.get("TVSHIFT_ONLY") else None


def video_frame(obs, ctx, pick, label):
    from PIL import Image, ImageDraw

    top = np.concatenate([np.asarray(Image.fromarray(obs[k]).resize((256, 256))) for k in ("image_full", "wrist_image_full")], 1)
    strip = np.concatenate([np.asarray(Image.fromarray(ctx[i][0]).resize((102, 102))) for i in pick], 1)
    strip = np.pad(strip, ((0, 0), (0, 512 - strip.shape[1]), (0, 0)))
    p = Image.fromarray(np.concatenate([top, strip], 0))
    d = ImageDraw.Draw(p)
    d.rectangle([0, 0, 512, 13], fill=(0, 0, 0))
    d.text((3, 1), label, fill=(255, 255, 255))
    d.text((3, 258), "context o^s", fill=(255, 255, 0))
    return np.asarray(p)


def combos():
    if ONLY is not None:
        out = [("seen", t, s, i, own) for t, s, i, own in TASKS]
        out += [("novel", t, s, i, th) for t, s, i, own in TASKS for th in ANGLES if th != own]
        out += [("sweep", t, s, i, th) for t, s, i, own in TASKS for th in SWEEP_ANGLES]
        return [c for c in out if tag_of(c) in ONLY]
    if os.environ.get("TVSHIFT_SWEEP"):
        return [("sweep", t, s, i, th) for t, s, i, own in TASKS if t in SWEEP_TASKS_ACTIVE for th in SWEEP_ANGLES]
    out = [("seen", t, s, i, own) for t, s, i, own in TASKS if t in TASKS_ACTIVE]
    out += [("novel", t, s, i, th) for t, s, i, own in TASKS if t in TASKS_ACTIVE for th in ANGLES if th != own]
    return out


def tag_of(c):
    return f"{c[0]}_t{c[1]}_{c[2]}_idx{c[3]}_theta{c[4]:03d}"


def summary_file(out, c):
    return Path(out) / tag_of(c) / f"{c[2]}_summary.json"


class EnvClient:
    def __init__(self, args, gpu):
        self.listener = Listener(("127.0.0.1", 0), authkey=AUTHKEY)
        port = self.listener.address[1]
        # hosts like hilo2 expose a single EGL device (id 0): always render there
        # ENV_WORKER_PYTHONPATH: e.g. a MuJoCo 3.3.2 wheel dir. The tvshift training data was rendered with
        # MuJoCo 3.3.2; MuJoCo 3.8.1 (the lingbot-va env) renders the libero_object floor scene much
        # darker (agentview mean 137.7 -> 102.5; spatial/goal unchanged).
        env = dict(os.environ, MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="0", CUDA_VISIBLE_DEVICES="0",
                   PYTHONPATH=os.environ.get("ENV_WORKER_PYTHONPATH", MUJOCO332))
        self.proc = subprocess.Popen(
            [args.env_python, ENV_WORKER, "--port", str(port), "--libero-root", args.libero_root,
             "--adapt-repo", args.adapt_repo], env=env)
        self.conn = self.listener.accept()

    def call(self, **msg):
        self.conn.send(msg)
        r = self.conn.recv()
        if BLANK_AGENTVIEW and "obs" in r:
            for k in ("image", "image_full"):
                if k in r["obs"]:
                    r["obs"][k] = np.zeros_like(r["obs"][k])
        return r

    def close(self):
        try:
            self.call(cmd="close")
        except Exception:
            pass
        self.proc.wait(timeout=60)


def load_policy(args):
    sys.path.insert(0, "/workspace/yujin/icwm/lerobot/src")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.icwm.modeling_icwm import ICWMPolicy

    # Checkpoints written by the tvshift training run have a config.json without the "type" key
    # that draccus needs to pick the policy class; add it in a scratch copy (weights and the
    # processors are still read from the original directory).
    import tempfile

    raw = json.loads((Path(args.ckpt) / "config.json").read_text())
    raw.setdefault("type", "icwm")
    scratch = tempfile.mkdtemp()
    (Path(scratch) / "config.json").write_text(json.dumps(raw))
    cfg = PreTrainedConfig.from_pretrained(scratch)
    cfg.device = "cuda"
    policy = ICWMPolicy.from_pretrained(args.ckpt, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.ckpt)
    return policy, pre, post, cfg


def to_chw(img):  # uint8 HWC -> float CHW in [0,1]
    import torch

    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0)


def probe(envc, obs, rng):
    """20 task-agnostic probing moves; returns [(o^s uint8 HWC, actions [5,7], o^e)]."""
    clips = []
    ee0 = obs["state"][:3].copy()
    for _ in range(N_PROBES):
        o_s = obs["image_full"]
        target = ee0 + rng.uniform(PROBE_LO - PROBE_REF, PROBE_HI - PROBE_REF)
        d = target - obs["state"][:3]
        d = d / max(np.linalg.norm(d), 1e-6)
        a = np.zeros(7, np.float32)
        a[:3] = PROBE_SCALE * d
        a[6] = -1.0  # gripper open
        acts = []
        for _ in range(PROBE_STEPS):
            r = envc.call(cmd="step", action=a)
            obs = r["obs"]
            acts.append(a.copy())
        clips.append((o_s, np.stack(acts), obs["image_full"]))
    return clips


def blank_clips(obs):
    """Context ablation: clips with no information -- black frames and a motionless arm (gripper open)."""
    black = np.zeros_like(obs["image_full"])
    a = np.zeros((PROBE_STEPS, 7), np.float32)
    a[:, 6] = -1.0
    return [(black, a, black)] * N_CONTEXT


def run_episode(policy, pre, post, cfg, envc, c, episode, rng, probe_env=None, no_context=False, history=False,
                empty_context=False, mixed=False, mixed_keep="both"):
    import torch

    proto, tid, suite, idx, theta = c
    kw = dict(cmd="reset", suite=suite, idx=idx, theta=theta, episode=episode, full=True)
    no_context = no_context or empty_context
    if no_context or history:
        r = envc.call(**kw)
    elif probe_env is None:
        r = envc.call(**kw)
    else:  # record the clips in another env (scene / camera), then execute in the task's own env
        ps, pi, pt = probe_env
        r = envc.call(cmd="reset", suite=ps, idx=pi, theta=pt, episode=episode, full=True)
    obs = r["obs"]
    if history:
        clips, prompt = [], r["prompt"]  # executed chunks of this episode, filled as it runs
    else:
        hist = []  # mixed: executed chunks of this episode, filled as it runs
        clips = blank_clips(obs) if no_context else probe(envc, obs, rng)
        r = envc.call(**kw)  # back to the identical initial state; only the clips carry over
        obs, prompt = r["obs"], r["prompt"]
    policy.reset()

    cams = [k for k, f in cfg.input_features.items() if f.type.value == "VISUAL"]
    full = {"observation.images.image": "image_full", "observation.images.wrist_image": "wrist_image_full"}
    done, t, frames = False, 0, []
    chunk_obs, chunk_acts = obs["image_full"], []
    while t < HORIZON and not done:
        if len(policy._action_queue) == 0:  # noqa: SLF001 -- a new chunk is needed: build the in-context batch
            if history or mixed:
                past = clips if history else hist
                if chunk_acts:  # the chunk just finished becomes the newest clip
                    acts = np.stack(chunk_acts)[:PROBE_STEPS]
                    if len(acts) < PROBE_STEPS:
                        acts = np.concatenate([acts, np.repeat(acts[-1:], PROBE_STEPS - len(acts), 0)])
                    past.append((chunk_obs, acts, obs["image_full"]))
                chunk_obs, chunk_acts = obs["image_full"], []
                past = past[-N_CONTEXT:]
                n_valid = len(past)
                hctx = (blank_clips(obs) + past)[-N_CONTEXT:]  # leading placeholders are dropped by the processor
                if history:
                    clips = past
                else:
                    hist = past
            if history:
                ctx = hctx
                pick = np.arange(N_CONTEXT)
            else:
                ctx = clips
                pick = rng.choice(len(clips), size=N_CONTEXT, replace=len(clips) < N_CONTEXT)
            raw = {k: to_chw(obs[full[k]]) for k in cams}
            raw["observation.state"] = torch.from_numpy(obs["state"])
            # clip tensors already carry the batch dim (the batch step only adds it to state/images)
            raw["observation.icwm_context.images_start"] = torch.stack([to_chw(ctx[i][0]) for i in pick])[None]
            raw["observation.icwm_context.images_end"] = torch.stack([to_chw(ctx[i][2]) for i in pick])[None]
            raw["observation.icwm_context.actions"] = torch.from_numpy(np.stack([ctx[i][1] for i in pick]))[None]
            if mixed:
                raw["observation.icwm_history.images_start"] = torch.stack([to_chw(h[0]) for h in hctx])[None]
                raw["observation.icwm_history.images_end"] = torch.stack([to_chw(h[2]) for h in hctx])[None]
                raw["observation.icwm_history.actions"] = torch.from_numpy(np.stack([h[1] for h in hctx]))[None]
                raw["observation.icwm_history.n_valid"] = torch.tensor([0 if mixed_keep == "explore" else n_valid])
                if mixed_keep == "history":
                    raw["observation.icwm_context.n_valid"] = torch.tensor([0])
            if history:
                raw["observation.icwm_context.n_valid"] = torch.tensor([n_valid])
            elif empty_context:  # no clip at all in the prompt (the processor drops all N placeholders)
                raw["observation.icwm_context.n_valid"] = torch.tensor([0])
            raw["task"] = [prompt]
            batch = pre(raw)
        else:
            batch = {}
        with torch.no_grad():
            a = policy.select_action(batch)
        a = post(a)[0].float().cpu().numpy()
        a = np.clip(a, -1.0, 1.0)
        a[6] = 1.0 if a[6] > 0 else -1.0  # LIBERO gripper is binary
        r = envc.call(cmd="step", action=a)
        chunk_acts.append(a.astype(np.float32))
        obs, done = r["obs"], r["done"]
        t += 1
        if SAVE_VIDEO:
            frames.append(video_frame(obs, ctx, pick, f"ICWM {tag_of(c)} ep{episode} step {t}"))
    if SAVE_VIDEO:
        import imageio

        Path(SAVE_VIDEO).mkdir(parents=True, exist_ok=True)
        imageio.mimsave(f"{SAVE_VIDEO}/{tag_of(c)}_ep{episode}_{'success' if done else 'fail'}.mp4", frames,
                        fps=20, macro_block_size=1)
    return bool(done), prompt


def worker_main(args):
    gpu = args.gpu
    policy, pre, post, cfg = load_policy(args)
    envc = EnvClient(args, gpu)
    rng = np.random.default_rng(1000 + args.bucket)
    mine = combos()[args.bucket::args.nbuckets]
    t0 = time.time()
    for c in mine:
        sf = summary_file(args.out, c)
        if sf.exists():
            continue
        wins, prompt = 0, ""
        for ep in range(args.episodes):
            probe_env = (c[2], c[3], farthest_angle(c[4])) if args.wrong_ctx_angle else None
            ok, prompt = run_episode(policy, pre, post, cfg, envc, c, ep, rng, probe_env=probe_env,
                                     no_context=args.no_context, history=args.history_context,
                                     empty_context=args.empty_context, mixed=args.mixed_context, mixed_keep=args.mixed_keep)
            wins += ok
            print(f"[{tag_of(c)}] ep {ep}: {'success' if ok else 'fail'} ({wins}/{ep + 1}) "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text(json.dumps({"benchmark": c[2], "tasks": [{
            "task_id": c[1], "prompt": prompt, "succ_num": float(wins), "total_num": float(args.episodes),
            "succ_rate": wins / args.episodes}]}))
    envc.close()


def ctx_units(args):
    return [(v, ep) for v in CTX_SWAP for ep in range(args.episodes)]


def ctx_worker_main(args):
    policy, pre, post, cfg = load_policy(args)
    envc = EnvClient(args, args.gpu)
    t0 = time.time()
    for v, ep in ctx_units(args)[args.bucket::args.nbuckets]:
        f = Path(args.out) / v / f"ep{ep:02d}.json"
        if f.exists():
            continue
        es, ei, et, ps, pi, pt = CTX_SWAP[v]
        rng = np.random.default_rng(ep)  # same probing targets / clip draws for every variant
        ok, prompt = run_episode(policy, pre, post, cfg, envc, ("ctx", 6, es, ei, et), ep, rng, (ps, pi, pt))
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"variant": v, "episode": ep, "success": ok, "prompt": prompt}))
        print(f"[{v}] ep {ep}: {'success' if ok else 'fail'} elapsed={time.time() - t0:.0f}s", flush=True)
    envc.close()


def ctx_parent_main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    procs = []
    gpus = args.gpus.split(",")
    for i, g in enumerate(gpus):
        cmd = [sys.executable, __file__, "--ctx-swap", "--worker", "--gpu", g, "--bucket", str(i), "--nbuckets", str(len(gpus))]
        for k in ("ckpt", "out", "episodes", "env_python", "libero_root", "adapt_repo"):
            cmd += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, TOKENIZERS_PARALLELISM="false", PYTHONPATH="/workspace/yujin/icwm/lerobot/src")
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(out / f"worker{g}.log", "a"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    rows = {}
    for v, ep in ctx_units(args):
        f = out / v / f"ep{ep:02d}.json"
        if not f.exists():
            print(f"INCOMPLETE: {v} ep {ep} missing", flush=True)
            sys.exit(1)
        rows.setdefault(v, []).append(json.loads(f.read_text())["success"])
    summ = {v: {"probe_env": CTX_SWAP[v][3:], "successes": sum(r), "episodes": len(r), "per_episode": r} for v, r in rows.items()}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    for v, d in summ.items():
        print(f"  {v:15s}: {d['successes']}/{d['episodes']}")


def parent_main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus.split(",")
    procs = []
    for i, g in enumerate(gpus):
        cmd = [sys.executable, __file__, "--worker", "--gpu", g, "--bucket", str(i), "--nbuckets", str(len(gpus))]
        cmd += ["--no-context"] if args.no_context else []
        cmd += ["--empty-context"] if args.empty_context else []
        cmd += ["--wrong-ctx-angle"] if args.wrong_ctx_angle else []
        cmd += ["--history-context"] if args.history_context else []
        cmd += ["--mixed-context", "--mixed-keep", args.mixed_keep] if args.mixed_context else []
        for k in ("ckpt", "out", "episodes", "env_python", "libero_root", "adapt_repo"):
            cmd += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, TOKENIZERS_PARALLELISM="false", PYTHONPATH="/workspace/yujin/icwm/lerobot/src")
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
    if missing:  # a worker died: no summary.json, so a polling driver relaunches and resumes
        print(f"INCOMPLETE: {len(missing)} combos missing, e.g. {missing[:3]}", flush=True)
        sys.exit(1)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    for p, (a, b) in tot.items():
        print(f"  {p:6s}: {a:.0f}/{b:.0f} = {100 * a / max(b, 1):.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrained_model dir (config + weights + processors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--env-python", default="/opt/conda/envs/lingbot-va/bin/python")
    ap.add_argument("--libero-root", default="/workspace/lerobot-libero")
    ap.add_argument("--adapt-repo", default="/workspace/yujin/lingbot-va-adapt")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--bucket", type=int, default=0)
    ap.add_argument("--nbuckets", type=int, default=1)
    ap.add_argument("--no-context", action="store_true", help="ablation: blank context clips, no probing")
    ap.add_argument("--empty-context", action="store_true",
                    help="ablation: no context clip at all in the prompt (n_valid=0), no probing")
    ap.add_argument("--wrong-ctx-angle", action="store_true",
                    help="ablation: probe the same scene with the camera at the farthest other trained angle")
    ap.add_argument("--history-context", action="store_true",
                    help="Reflective-VLA style context: the last N executed chunks of this episode, no probing")
    ap.add_argument("--mixed-context", action="store_true",
                    help="context_source=mixed models: probing clips + the last N executed chunks of this episode")
    ap.add_argument("--mixed-keep", choices=("both", "explore", "history"), default="both",
                    help="with --mixed-context: keep both segments, or only one of them")
    ap.add_argument("--ctx-swap", action="store_true", help="context-swap study on object idx0 @270 (CTX_SWAP)")
    a = ap.parse_args()
    if a.ctx_swap:
        ctx_worker_main(a) if a.worker else ctx_parent_main(a)
    else:
        worker_main(a) if a.worker else parent_main(a)
