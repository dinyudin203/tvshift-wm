"""Decode the tvshift LeRobot v3.0 datasets once into uint8 memmaps (the "frame cache").

The source videos are 512x512 and the triplet sampler touches 2K-1 frames per item, so
decoding on the fly would dominate training. Frames are resized to the pi0.5 input size
(224x224) and stored as one flat [N, 224, 224, 3] memmap per camera, indexed by the
dataset-global frame index. Everything else (state, action, episode bounds, task string)
goes in a small npz.

    python -m reflective_vla.data.build_cache --out /workspace/yujin/data/reflective_tvshift_cache
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/workspace/yujin/data/libero_tvshift/lerobot_v30")
# (name, dataset dir, trained angle) -- the 4 training datasets, 2 tasks each.
DATASETS = [
    ("spatial_000", "libero_spatial_theta000_no_noops_lerobot", 0),
    ("goal_045", "libero_goal_theta045_no_noops_lerobot", 45),
    ("goal_345", "libero_goal_theta345_no_noops_lerobot", 345),
    ("object_270", "libero_object_theta270_no_noops_lerobot", 270),
]
CAMS = ["image", "wrist_image"]
RES = 224


def decode_file(path: Path, n_expected: int, out: np.memmap, offset: int) -> int:
    """Decode every frame of ``path`` (resized) into out[offset:offset+n]."""
    n = 0
    with av.open(str(path)) as c:
        for frame in c.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            out[offset + n] = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_AREA)
            n += 1
    return n


def build_one(name: str, dname: str, angle: int, out_dir: Path) -> dict:
    d = ROOT / dname
    info = json.loads((d / "meta/info.json").read_text())
    n_total = info["total_frames"]
    data = pq.read_table(sorted(glob.glob(str(d / "data/**/*.parquet"), recursive=True))[0]).to_pandas() \
        if len(glob.glob(str(d / "data/**/*.parquet"), recursive=True)) == 1 else None
    if data is None:
        import pandas as pd
        data = pd.concat([pq.read_table(f).to_pandas()
                          for f in sorted(glob.glob(str(d / "data/**/*.parquet"), recursive=True))])
    data = data.sort_values("index").reset_index(drop=True)
    assert len(data) == n_total and (data["index"].values == np.arange(n_total)).all()

    eps = pq.read_table(sorted(glob.glob(str(d / "meta/episodes/**/*.parquet"), recursive=True))[0]).to_pandas()
    tasks = {}
    for line in (d / "meta/tasks.jsonl").read_text().splitlines():
        t = json.loads(line)
        tasks[t["task_index"]] = t["task"]

    fps = info["fps"]
    for cam in CAMS:
        key = f"observation.images.{cam}"
        mm = np.lib.format.open_memmap(out_dir / f"{name}_{cam}.npy", mode="w+",
                                       dtype=np.uint8, shape=(n_total, RES, RES, 3))
        # v3.0 packs episodes back to back into video files; a file's frames are the
        # episodes' frames in order, so place each file at its first episode's start.
        files = {}
        for _, e in eps.iterrows():
            f = (int(e[f"videos/{key}/chunk_index"]), int(e[f"videos/{key}/file_index"]))
            files.setdefault(f, []).append(e)
        for (ch, fi), es in files.items():
            first = min(es, key=lambda e: e["dataset_from_index"])
            p = d / info["video_path"].format(video_key=key, chunk_index=ch, file_index=fi)
            got = decode_file(p, 0, mm, int(first["dataset_from_index"]))
            want = sum(int(e["length"]) for e in es)
            assert got == want, f"{p}: decoded {got} frames, episodes need {want}"
        mm.flush()

    np.savez(out_dir / f"{name}_meta.npz",
             state=np.stack(data["observation.state"].values).astype(np.float32),
             action=np.stack(data["action"].values).astype(np.float32),
             episode_index=data["episode_index"].values.astype(np.int64),
             frame_index=data["frame_index"].values.astype(np.int64),
             task_index=data["task_index"].values.astype(np.int64))
    (out_dir / f"{name}_tasks.json").write_text(json.dumps({"angle": angle, "tasks": tasks, "fps": fps,
                                                              "n_frames": n_total}))
    return {"name": name, "frames": n_total, "episodes": len(eps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, dname, angle in DATASETS:
        print(build_one(name, dname, angle, out), flush=True)


if __name__ == "__main__":
    main()
