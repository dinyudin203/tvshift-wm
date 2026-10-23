"""
Stage B: render_shard.py 가 만든 에피소드 shard 들을 theta 별 LeRobot v3.0 데이터셋으로 묶는다.

출력 레이아웃 (yuanty/LIBERO-fastwam 의 lerobot_v30 와 동일한 스키마):
  <out>/libero_spatial_theta{TTT}_no_noops_lerobot/
    meta/info.json
    meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet
    meta/stats.json
    data/chunk-000/file-{FFF}.parquet
    videos/observation.images.image/chunk-000/file-{FFF}.mp4
    videos/observation.images.wrist_image/chunk-000/file-{FFF}.mp4

usage:
  python pack_lerobot.py --shards .../shards --out .../lerobot_v30 --theta 30
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 20
CHUNKS_SIZE = 1000
DATA_FILE_MB = 100
VIDEO_FILE_MB = 200
VIDEO_KEYS = ["observation.images.image", "observation.images.wrist_image"]
VECTOR_KEYS = [
    "observation.state",
    "observation.states.ee_state",
    "observation.states.joint_state",
    "observation.states.gripper_state",
    "action",
]
SCALAR_KEYS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]


def build_features(res):
    def video_feat():
        return {
            "dtype": "video",
            "shape": [res, res, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": res,
                "video.width": res,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    def vec(shape, motors):
        return {"dtype": "float32", "shape": [shape], "names": {"motors": motors}, "fps": FPS}

    xyzrpy = ["x", "y", "z", "roll", "pitch", "yaw"]
    return {
        "observation.images.image": video_feat(),
        "observation.images.wrist_image": video_feat(),
        "observation.state": vec(8, xyzrpy + ["gripper", "gripper"]),
        "observation.states.ee_state": vec(6, xyzrpy),
        "observation.states.joint_state": vec(7, [f"joint_{i}" for i in range(7)]),
        "observation.states.gripper_state": vec(2, ["gripper", "gripper"]),
        "action": vec(7, xyzrpy + ["gripper"]),
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": FPS},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
    }


# ------------------------------------------------------------------ utilities
def concat_mp4(clip_paths, out_path):
    """all-intra h264 클립들을 concat demuxer 로 무손실 이어붙인다."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for p in clip_paths:
            fh.write(f"file '{os.path.abspath(p)}'\n")
        list_path = fh.name
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
           "-i", list_path, "-c", "copy", "-fflags", "+genpts", out_path]
    subprocess.run(cmd, check=True)
    os.unlink(list_path)


def video_stats(path, total, max_samples=8):
    """all-intra 클립에서 max_samples 장만 균등 샘플링해 채널별 통계를 낸다."""
    import av

    step = max(1, total // max_samples)
    frames = []
    container = av.open(path)
    for i, frame in enumerate(container.decode(video=0)):
        if i % step == 0:
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= max_samples:
                break
    container.close()
    arr = np.stack(frames).astype(np.float32) / 255.0  # (n,h,w,3)
    per_c = arr.reshape(-1, 3)

    def nest(v):
        return [[[float(x)]] for x in v]

    return {
        "min": nest(per_c.min(0)), "max": nest(per_c.max(0)),
        "mean": nest(per_c.mean(0)), "std": nest(per_c.std(0)),
        "count": [int(total)],
    }


def array_stats(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return {
        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
        "count": [int(a.shape[0])],
    }


def aggregate(stats_list):
    """에피소드 통계들을 데이터셋 전체 통계로 합친다 (min/max/가중평균/합산분산)."""
    counts = np.array([s["count"][0] for s in stats_list], dtype=np.float64)
    total = counts.sum()
    means = np.array([np.asarray(s["mean"], dtype=np.float64) for s in stats_list])
    stds = np.array([np.asarray(s["std"], dtype=np.float64) for s in stats_list])
    mins = np.array([np.asarray(s["min"], dtype=np.float64) for s in stats_list])
    maxs = np.array([np.asarray(s["max"], dtype=np.float64) for s in stats_list])
    w = counts.reshape(-1, *([1] * (means.ndim - 1)))
    mean = (means * w).sum(0) / total
    var = ((stds ** 2 + (means - mean) ** 2) * w).sum(0) / total
    return {
        "min": mins.min(0).tolist(), "max": maxs.max(0).tolist(),
        "mean": mean.tolist(), "std": np.sqrt(var).tolist(), "count": [int(total)],
    }


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--theta", type=int, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or f"{args.suite}_theta{args.theta:03d}_no_noops_lerobot"
    root = os.path.join(args.out, name)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)

    manifests = []
    for p in sorted(glob.glob(os.path.join(args.shards, "_done", "*.json"))):
        with open(p) as fh:
            manifests.append(json.load(fh))
    manifests.sort(key=lambda m: (m["task"], m["demo"]))
    assert manifests, "no shards found"

    # ---- tasks.parquet
    languages = []
    for m in manifests:
        if m["language"] not in languages:
            languages.append(m["language"])
    languages_sorted = sorted(languages)
    task_index = {lang: i for i, lang in enumerate(languages_sorted)}
    tasks_df = pd.DataFrame({"task_index": [task_index[l] for l in languages_sorted]}, index=languages_sorted)
    os.makedirs(os.path.join(root, "meta"), exist_ok=True)
    tasks_df.to_parquet(os.path.join(root, "meta", "tasks.parquet"))
    # scripts/precompute_text_embeds.py 는 v2.1 형식의 tasks.jsonl 을 읽으므로 같이 써 둔다
    with open(os.path.join(root, "meta", "tasks.jsonl"), "w", encoding="utf-8") as fh:
        for lang in languages_sorted:
            fh.write(json.dumps({"task_index": task_index[lang], "task": lang}) + "\n")

    features = build_features(args.res)

    # ---- 에피소드 순회
    ep_rows = []
    data_rows = []
    data_file_index = 0
    data_rows_start = 0          # 현재 data 파일이 담고 있는 첫 글로벌 index
    video_state = {k: {"file_index": 0, "frames": 0, "clips": [], "bytes": 0} for k in VIDEO_KEYS}
    global_index = 0
    all_stats = {k: [] for k in VECTOR_KEYS + SCALAR_KEYS + VIDEO_KEYS}

    def flush_data():
        nonlocal data_rows, data_file_index, data_rows_start
        if not data_rows:
            return
        df = pd.concat(data_rows, ignore_index=True)
        path = os.path.join(root, "data", "chunk-000", f"file-{data_file_index:03d}.parquet")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        data_rows = []
        data_file_index += 1
        data_rows_start = global_index

    def flush_video(key):
        st = video_state[key]
        if not st["clips"]:
            return
        path = os.path.join(root, "videos", key, "chunk-000", f"file-{st['file_index']:03d}.mp4")
        concat_mp4(st["clips"], path)
        st["clips"] = []
        st["file_index"] += 1
        st["frames"] = 0
        st["bytes"] = 0

    for ep_idx, m in enumerate(manifests):
        stem = m["stem"]
        npz = np.load(os.path.join(args.shards, "frames", f"{stem}.npz"))
        length = int(m["length"])
        clip_paths = {
            "observation.images.image": os.path.join(args.shards, f"theta{args.theta:03d}", f"{stem}.mp4"),
            "observation.images.wrist_image": os.path.join(args.shards, "wrist", f"{stem}.mp4"),
        }

        row = {}
        for k in VECTOR_KEYS:
            arr = npz[k].astype(np.float32)
            assert len(arr) == length, f"{stem} {k} {len(arr)} != {length}"
            row[k] = list(arr)
            all_stats[k].append(array_stats(arr))
        row["timestamp"] = np.arange(length, dtype=np.float32) / FPS
        row["frame_index"] = np.arange(length, dtype=np.int64)
        row["episode_index"] = np.full(length, ep_idx, dtype=np.int64)
        row["index"] = np.arange(global_index, global_index + length, dtype=np.int64)
        row["task_index"] = np.full(length, task_index[m["language"]], dtype=np.int64)
        for k in SCALAR_KEYS:
            all_stats[k].append(array_stats(np.asarray(row[k])))
        data_rows.append(pd.DataFrame(row))

        ep_row = {
            "episode_index": ep_idx,
            "data/chunk_index": 0,
            "data/file_index": data_file_index,
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + length,
            "tasks": np.array([m["language"]], dtype=object),
            "length": length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for key in VIDEO_KEYS:
            st = video_state[key]
            ep_row[f"videos/{key}/chunk_index"] = 0
            ep_row[f"videos/{key}/file_index"] = st["file_index"]
            ep_row[f"videos/{key}/from_timestamp"] = st["frames"] / FPS
            ep_row[f"videos/{key}/to_timestamp"] = (st["frames"] + length) / FPS
            st["clips"].append(clip_paths[key])
            st["frames"] += length
            st["bytes"] += os.path.getsize(clip_paths[key])
            all_stats[key].append(video_stats(clip_paths[key], length))
        ep_rows.append(ep_row)

        global_index += length

        # 파일 롤오버
        approx_data_mb = sum(len(d) for d in data_rows) * 200 / 1e6
        if approx_data_mb > DATA_FILE_MB:
            flush_data()
        for key in VIDEO_KEYS:
            if video_state[key]["bytes"] > VIDEO_FILE_MB * 1e6:
                flush_video(key)

        if (ep_idx + 1) % 50 == 0:
            print(f"[{name}] {ep_idx+1}/{len(manifests)} episodes", flush=True)

    flush_data()
    for key in VIDEO_KEYS:
        flush_video(key)

    # ---- meta/episodes
    ep_df = pd.DataFrame(ep_rows)
    # stats 컬럼은 중첩 리스트가 깨지지 않도록 object Series 로 대입한다
    for key in VECTOR_KEYS + SCALAR_KEYS + VIDEO_KEYS:
        for field in ["min", "max", "mean", "std", "count"]:
            if field == "count":
                values = [np.array(s[field], dtype=np.int64) for s in all_stats[key]]
            elif key in VIDEO_KEYS:
                # (3,1,1) 중첩 구조는 파이썬 리스트로 둬야 parquet 이 list<list<list<double>>> 로 받는다
                values = [s[field] for s in all_stats[key]]
            else:
                values = [np.array(s[field], dtype=np.float64) for s in all_stats[key]]
            ep_df[f"stats/{key}/{field}"] = pd.Series(values, dtype=object)
    ep_path = os.path.join(root, "meta", "episodes", "chunk-000", "file-000.parquet")
    os.makedirs(os.path.dirname(ep_path), exist_ok=True)
    ep_df.to_parquet(ep_path, index=False)

    # ---- meta/stats.json
    stats_json = {k: aggregate(v) for k, v in all_stats.items()}
    with open(os.path.join(root, "meta", "stats.json"), "w") as fh:
        json.dump(stats_json, fh, indent=1)

    # ---- meta/info.json
    info = {
        "codebase_version": "v3.0",
        "robot_type": "franka",
        "total_episodes": len(manifests),
        "total_frames": int(global_index),
        "total_tasks": len(languages_sorted),
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{len(manifests)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "data_files_size_in_mb": DATA_FILE_MB,
        "video_files_size_in_mb": VIDEO_FILE_MB,
    }
    with open(os.path.join(root, "meta", "info.json"), "w") as fh:
        json.dump(info, fh, indent=4)

    print(f"[{name}] done: {len(manifests)} episodes, {global_index} frames -> {root}", flush=True)


if __name__ == "__main__":
    main()
