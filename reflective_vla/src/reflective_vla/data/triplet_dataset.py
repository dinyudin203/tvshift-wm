"""Triplet sampler for Reflective VLA on the tvshift cache (paper Appendix C.1).

One item = K frames of one episode, oldest first, the query frame last:

    t_{K-1} = t                                   (query / current observation)
    t_{k-1} = t_k - (C + g_k),  g_k ~ U[0, gap]   (random gap: no fixed-stride shortcut)

Frame k contributes its observation O_k, the chunk it "executed" A_k = action[t_k : t_k+C]
(which is also its flow-matching target) and, for k < K-1, the consequence
O'_k = O_{t_k + C}. Frames that would fall before the episode start are masked
(``observation.frame_mask`` False, images zero) -- they are neither context nor target.
Actions stay in the dataset's native *delta* EEF space, quantile-normalised to [-1, 1].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .build_cache import CAMS, DATASETS

STATE_KEY = "observation.state"
CAM_KEYS = [f"observation.images.{c}" for c in CAMS]


class TripletDataset(Dataset):
    def __init__(self, cache_dir, *, K=8, C=10, gap_max=15, tokenizer=None, max_len=48,
                 stats=None, train=True, items_per_epoch=None, seed=0):
        self.dir = Path(cache_dir)
        self.K, self.C, self.gap_max, self.train = K, C, gap_max, train
        self.items_per_epoch = items_per_epoch
        self.rng_seed = seed
        self.names = [n for n, _, _ in DATASETS]
        self.meta, self.imgs, self.tasks, self.ep_bounds = {}, {}, {}, {}
        self._index = []  # (dataset, global frame idx) of every valid query timestep
        for name in self.names:
            m = np.load(self.dir / f"{name}_meta.npz")
            t = json.loads((self.dir / f"{name}_tasks.json").read_text())
            self.meta[name] = {k: m[k] for k in m.files}
            self.tasks[name] = {int(k): v for k, v in t["tasks"].items()}
            self.angles = getattr(self, "angles", {}); self.angles[name] = int(t["angle"])
            ep = self.meta[name]["episode_index"]
            starts = np.r_[0, np.flatnonzero(np.diff(ep)) + 1]
            ends = np.r_[starts[1:], len(ep)]
            self.ep_bounds[name] = (starts, ends)
            for s, e in zip(starts, ends):
                self._index += [(name, i) for i in range(s, e)]
        self._tok = {}
        self.tokenizer, self.max_len = tokenizer, max_len
        self.stats = stats if stats is not None else self.compute_stats()

    # ------------------------------------------------------------------ stats
    def compute_stats(self):
        st = np.concatenate([m["state"] for m in self.meta.values()])
        ac = np.concatenate([m["action"] for m in self.meta.values()])
        q = lambda x: {"q01": np.quantile(x, 0.01, 0), "q99": np.quantile(x, 0.99, 0)}  # noqa: E731
        return {"state": q(st), "action": q(ac)}

    @staticmethod
    def _norm(x, s):
        lo, hi = s["q01"], s["q99"]
        return np.clip(2 * (x - lo) / np.maximum(hi - lo, 1e-6) - 1, -1.5, 1.5).astype(np.float32)

    def _images(self, name):
        if name not in self.imgs:
            self.imgs[name] = [np.load(self.dir / f"{name}_{c}.npy", mmap_mode="r") for c in CAMS]
        return self.imgs[name]

    def _tokens(self, text):
        if text not in self._tok:
            out = self.tokenizer(text, max_length=self.max_len, padding="max_length",
                                 truncation=True, return_tensors="pt")
            self._tok[text] = (out["input_ids"][0], out["attention_mask"][0].bool())
        return self._tok[text]

    def __len__(self):
        return self.items_per_epoch or len(self._index)

    # ------------------------------------------------------------------ item
    def __getitem__(self, i):
        rng = np.random.default_rng((self.rng_seed, i)) if not self.train else np.random.default_rng()
        name, q = self._index[i % len(self._index)] if self.items_per_epoch is None \
            else self._index[rng.integers(len(self._index))]
        K, C = self.K, self.C
        meta = self.meta[name]
        starts, ends = self.ep_bounds[name]
        e = np.searchsorted(starts, q, side="right") - 1
        s0, e0 = int(starts[e]), int(ends[e])

        # frame times (global idx), oldest first
        gaps = rng.integers(0, self.gap_max + 1, size=K - 1) if self.train else np.zeros(K - 1, int)
        t = [q]
        for g in gaps:
            t.append(t[-1] - (C + int(g)))
        t = t[::-1]
        valid = np.array([ti >= s0 for ti in t])

        cams = self._images(name)
        zero = np.zeros((224, 224, 3), np.uint8)
        out = {}
        for key, arr in zip(CAM_KEYS, cams):
            out[key] = torch.from_numpy(np.stack([arr[ti] if v else zero for ti, v in zip(t, valid)])).permute(0, 3, 1, 2)
            cons_key = key.replace("observation.images", "observation.consequence_images")
            cons = []
            for ti, v in zip(t[:-1], valid[:-1]):
                tc = min(ti + C, e0 - 1)
                cons.append(arr[tc] if v else zero)
            # K=1 (reactive pi0.5): no context frame, so no consequence frame either
            cons = np.stack(cons) if cons else np.zeros((0, *zero.shape), np.uint8)
            out[cons_key] = torch.from_numpy(cons).permute(0, 3, 1, 2)
            out[f"{key}_is_pad"] = torch.from_numpy(~valid)
        out[STATE_KEY] = torch.from_numpy(np.stack(
            [self._norm(meta["state"][ti if v else s0], self.stats["state"]) for ti, v in zip(t, valid)]))
        cons_state = [self._norm(meta["state"][min(ti + C, e0 - 1) if v else s0], self.stats["state"])
                      for ti, v in zip(t[:-1], valid[:-1])]
        out["observation.consequence_state"] = torch.from_numpy(
            np.stack(cons_state) if cons_state else np.zeros((0, meta["state"].shape[1]), np.float32))

        acts, pads = [], []
        for ti, v in zip(t, valid):
            idx = np.arange(ti, ti + C)
            pad = (idx >= e0) | (not v)
            idx = np.clip(idx, s0, e0 - 1)
            acts.append(self._norm(meta["action"][idx], self.stats["action"]))
            pads.append(pad)
        acts = np.stack(acts)  # [K, C, A]
        out["action"] = torch.from_numpy(acts)
        out["action_is_pad"] = torch.from_numpy(np.stack(pads))
        out["observation.context_actions"] = out["action"][:-1].clone()
        out["observation.frame_mask"] = torch.from_numpy(valid)
        out["observation.temporal_index"] = torch.from_numpy((q - np.array(t)).astype(np.float32))  # steps ago, oldest largest
        prompt = self.tasks[name][int(meta["task_index"][q])]
        ids, mask = self._tokens(prompt)
        out["observation.language.tokens"] = ids
        out["observation.language.attention_mask"] = mask
        out["task"] = prompt
        out["angle"] = torch.tensor(self.angles[name])
        return out
