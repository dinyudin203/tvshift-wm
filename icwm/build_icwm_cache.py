"""Decode the 4 tvshift LeRobot datasets once into full-resolution uint8 memmaps.

ICWM samples ~36 video frames per training item (query + 5 clips x (start, end, 5 action reads)),
each one a keyframe seek + decode, which is what leaves the GPUs idle. The cache stores every
frame at the native 512x512 (so the model input is bit-identical to the on-the-fly path) as
[N, 512, 512, 3] uint8, indexed by the dataset-global frame index.

    python build_icwm_cache.py --out /workspace/yujin/data/icwm_tvshift_cache
"""
import argparse
import glob
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/workspace/yujin/data/libero_tvshift/lerobot_v30")
NAMES = ["libero_spatial_theta000_no_noops_lerobot", "libero_goal_theta045_no_noops_lerobot",
         "libero_goal_theta345_no_noops_lerobot", "libero_object_theta270_no_noops_lerobot"]
CAMS = ["observation.images.image", "observation.images.wrist_image"]


def build(name, out_dir, root=ROOT):
    d = root / name
    info = json.loads((d / "meta/info.json").read_text())
    n = info["total_frames"]
    eps = pq.read_table(sorted(glob.glob(str(d / "meta/episodes/**/*.parquet"), recursive=True))[0]).to_pandas()
    for cam in CAMS:
        path = out_dir / f"{name}__{cam}.npy"
        if (out_dir / f"{name}__{cam}.done").exists():
            print(f"{name} {cam}: already built", flush=True)
            continue
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(n, 512, 512, 3))
        files = {}
        for _, e in eps.iterrows():
            files.setdefault((int(e[f"videos/{cam}/chunk_index"]), int(e[f"videos/{cam}/file_index"])), []).append(e)
        for (ch, fi), es in files.items():
            first = min(es, key=lambda e: e["dataset_from_index"])
            vp = d / info["video_path"].format(video_key=cam, chunk_index=ch, file_index=fi)
            pos = int(first["dataset_from_index"])
            got = 0
            with av.open(str(vp)) as c:
                for frame in c.decode(video=0):  # same conversion as lerobot's pyav decoder
                    mm[pos + got] = frame.to_ndarray(format="rgb24")
                    got += 1
            assert got == sum(int(e["length"]) for e in es), f"{vp}: decoded {got} frames"
        mm.flush()
        (out_dir / f"{name}__{cam}.done").write_text("ok")
        print(f"{name} {cam}: {n} frames", flush=True)
    (out_dir / f"{name}__meta.json").write_text(json.dumps({"n_frames": n, "fps": info["fps"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(ROOT), help="parent dir of the datasets (default: tvshift)")
    ap.add_argument("--names", default="", help="comma-separated dataset names (default: the 4 tvshift ones)")
    ap.add_argument("--only", default="", help="build just this dataset (parallel runs); skips the DONE marker")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.names.split(",") if a.names else NAMES
    for n in ([a.only] if a.only else names):
        build(n, out, Path(a.root))
    if not a.only:
        (out / "DONE").write_text("ok")
