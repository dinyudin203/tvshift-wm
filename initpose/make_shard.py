"""
Stage A (initial pose): roll out the GT demos of one task from a fixed perturbed home
and write episode shards in the libero_rot layout, so FastWAM/libero_rot/pack_lerobot.py
packs them unchanged (agentview clips go to theta000/, pack with --theta 0).

usage:
  MUJOCO_GL=egl python make_shard.py --suite libero_spatial --task 2 --mag 0.3 --seed 0 \
      --out /workspace/yujin/data/libero_initpose/shards/libero_spatial_initpose_dq030
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/yujin/FastWAM/libero_rot")

import cv2
import h5py
import imageio
import numpy as np

import initpose
import replay
from render_shard import is_noop, write_mp4  # FastWAM/libero_rot


def label(img, text, color=(255, 255, 255)):
    img = img.copy()
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def pad(x, n):
    return np.concatenate([x, np.repeat(x[-1:], n - len(x), 0)]) if len(x) < n else x


def comparison_video(path, info, fr, gt, size=256):
    """GT demo (top) vs perturbed rollout (bottom), each agentview | wrist."""
    rs = lambda xs: np.stack([cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA) for x in xs])
    T = info["n_pre_hold"] + info["t_merge"]
    n = max(len(fr["agentview"]), len(gt["agentview"]))
    gt_av, gt_wr = pad(rs(gt["agentview"]), n), pad(rs(gt["wrist"]), n)
    pa_av, pa_wr = pad(rs(fr["agentview"]), n), pad(rs(fr["wrist"]), n)
    vid = []
    for k in range(n):
        phase = "approach (warped)" if k < T else "merged -> GT"
        top = np.concatenate([label(gt_av[k], f"GT demo  t={min(k, len(gt['agentview']) - 1)}"), gt_wr[k]], axis=1)
        bot = np.concatenate(
            [label(pa_av[k], f"|dq|={info['dq_norm']:.2f}  {phase}", (255, 230, 0)), pa_wr[k]], axis=1)
        vid.append(np.concatenate([top, np.full((4, top.shape[1], 3), 255, np.uint8), bot], axis=0))
    imageio.mimsave(path, np.stack(vid), fps=20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--mag", type=float, required=True,
                    help="signed perturbation along the seeded direction (rad); 0 = default (GT as is)")
    ap.add_argument("--seed", type=int, required=True, help="direction seed of the condition")
    ap.add_argument("--max-track-mm", type=float, default=1e9,
                    help="reject rollouts whose post-merge max tracking error exceeds this (mm)")
    ap.add_argument("--retries", type=int, default=0, help="extra attempts per demo when rejected")
    ap.add_argument("--demo-start", type=int, default=0)
    ap.add_argument("--demo-end", type=int, default=10**9)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--preview", type=int, default=1, help="write GT-vs-perturbed videos for the first N demos")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    q0 = initpose.condition_q0(args.mag, args.seed)
    t, _, h5 = replay.task_info(args.suite, args.task)
    with h5py.File(h5, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    demo_keys = demo_keys[args.demo_start : args.demo_end]

    t_start = time.time()
    for i, demo_key in enumerate(demo_keys):
        demo_idx = int(demo_key.split("_")[1])
        stem = f"task{args.task:02d}_demo{demo_idx:03d}"
        done_marker = os.path.join(args.out, "_done", f"{stem}.json")
        if os.path.exists(done_marker):
            print(f"[skip] {stem}", flush=True)
            continue
        preview = args.mag != 0 and i < args.preview
        # rollouts are not reproducible (contact-rich steps differ run to run), so retry until clean
        for attempt in range(args.retries + 1):
            if args.mag == 0:
                info, table, fr, gt = initpose.gt_episode(args.suite, args.task, demo_key, res=args.res)
            else:
                info, table, fr, gt = initpose.generate(args.suite, args.task, demo_key, q0, res=args.res,
                                                        render_gt=preview)
            track_max = info.get("track_err_mm_after_merge_mean_max", [0.0, 0.0])[1]
            if info["valid"] and track_max <= args.max_track_mm:
                break
            print(f"[retry] {stem} attempt={attempt + 1} success={info['success']} "
                  f"contacts={info.get('approach_contacts')} track_max={track_max:.1f}mm", flush=True)
        info["attempts"] = attempt + 1
        info["valid_raw"] = info["valid"]
        info["valid"] = bool(info["valid"] and track_max <= args.max_track_mm)
        os.makedirs(os.path.join(args.out, "_info"), exist_ok=True)
        with open(os.path.join(args.out, "_info", f"{stem}.json"), "w") as fh:
            json.dump(info, fh, indent=1)
        if preview:
            os.makedirs(os.path.join(args.out, "preview"), exist_ok=True)
            comparison_video(os.path.join(args.out, "preview", f"{stem}.mp4"), info, fr, gt)
        if not info["valid"]:
            print(f"[drop-invalid] {stem} success={info['success']} start={info.get('start_contacts')} "
                  f"approach={info.get('approach_contacts')}", flush=True)
            continue

        # no-op 프레임 제거 (openvla 방식)
        keep, prev = [], None
        for k, a in enumerate(table["action"]):
            if not is_noop(a, prev):
                keep.append(k)
            prev = a
        idx = np.array(keep)
        frame_table = {
            "observation.state": np.concatenate([table["ee_state"][idx], table["gripper_state"][idx]], axis=1),
            "observation.states.ee_state": table["ee_state"][idx],
            "observation.states.joint_state": table["joint_state"][idx],
            "observation.states.gripper_state": table["gripper_state"][idx],
            "action": table["action"][idx],
        }
        write_mp4(os.path.join(args.out, "wrist", f"{stem}.mp4"), [fr["wrist"][k] for k in keep], args.fps)
        write_mp4(os.path.join(args.out, "theta000", f"{stem}.mp4"), [fr["agentview"][k] for k in keep], args.fps)
        npz_path = os.path.join(args.out, "frames", f"{stem}.npz")
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(npz_path, **frame_table)

        os.makedirs(os.path.dirname(done_marker), exist_ok=True)
        with open(done_marker, "w") as fh:
            json.dump(
                {
                    "task": args.task,
                    "demo": demo_idx,
                    "stem": stem,
                    "language": t.language,
                    "length": int(len(keep)),
                    "raw_length": int(len(table["action"])),
                    "init_qpos": q0.tolist(),
                    "dq_norm": args.mag,
                    "dq_seed": args.seed,
                    "res": args.res,
                    "fps": args.fps,
                },
                fh,
            )
        print(f"[ok] {stem} kept={len(keep)}/{len(table['action'])} shift={info.get('start_eef_shift_cm', 0):.1f}cm "
              f"elapsed={time.time()-t_start:.0f}s", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
