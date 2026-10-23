"""Frame cache for ICWM training.

`LeRobotDataset` decodes video on every access (keyframe seek + decode). ICWM touches ~36 frames
per item, which starves the GPUs. When `ICWM_FRAME_CACHE=<dir>` is set (see
`build_icwm_cache.py`), `DatasetReader._query_videos` is replaced by a memmap lookup that returns
the very same frames the decoder would (native 512x512 uint8; float32/255 when the dataset was
built without `return_uint8`). Install it in the main process before DataLoader workers fork.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

_MM: dict[tuple[str, str], np.ndarray] = {}
_INSTALLED = False


def cache_dir() -> Path | None:
    d = os.environ.get("ICWM_FRAME_CACHE", "")
    return Path(d) if d else None


def frames(dataset_name: str, cam: str) -> np.ndarray:
    key = (dataset_name, cam)
    if key not in _MM:
        _MM[key] = np.load(cache_dir() / f"{dataset_name}__{cam}.npy", mmap_mode="r")
    return _MM[key]


def available(dataset_name: str) -> bool:
    d = cache_dir()
    return d is not None and (d / "DONE").exists() and (d / f"{dataset_name}__meta.json").exists()


def install() -> bool:
    """Patch DatasetReader._query_videos once. Returns True when the cache is active."""
    global _INSTALLED
    if cache_dir() is None or not (cache_dir() / "DONE").exists():
        return False
    if _INSTALLED:
        return True
    from lerobot.datasets.dataset_reader import DatasetReader

    original = DatasetReader._query_videos

    def cached_query_videos(self, query_timestamps, ep_idx):
        name = Path(self.root).name
        if not available(name):
            return original(self, query_timestamps, ep_idx)
        ep = self._meta.episodes[ep_idx]
        base, fps = int(ep["dataset_from_index"]), self._meta.fps
        out = {}
        for key, ts in query_timestamps.items():
            mm = frames(name, key)
            idx = [base + int(round(t * fps)) for t in ts]
            fr = torch.from_numpy(np.stack([mm[i] for i in idx])).permute(0, 3, 1, 2)  # uint8 [T,C,H,W]
            if not self._return_uint8:
                fr = fr.float() / 255.0
            out[key] = fr.squeeze(0)
        return out

    DatasetReader._query_videos = cached_query_videos
    _INSTALLED = True
    return True
