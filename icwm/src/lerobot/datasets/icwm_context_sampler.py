#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""In-context training data pipeline for ICWM (arXiv 2606.26025), paper Sec 4.1.

The paper trains on samples of the form

    [ N interaction clips T = {(o^s_i, a_i, o^e_i)} ]  +  [ o_t, l ]  ->  a_t

where the N clips are "randomly sampled from a pool of interaction segments collected
across ALL training trajectories and viewpoints" (i.e. the pool is *not* tied to the
episode/viewpoint the (o_t, l, a_t) query comes from -- diversity of psi across the pool is
the point: it is what teaches the backbone to actually use Psi(T) instead of ignoring it).

This module provides:
  * `ICWMInteractionSegmentPool`: builds that pool once, up front, from one or more
    `LeRobotDataset`s (e.g. one dataset per rendered camera azimuth).
  * `ICWMContextSampledDataset`: a thin `torch.utils.data.Dataset` wrapper around a "task"
    dataset (the one supplying the real (o_t, l, a_t) queries) that, for every `__getitem__`,
    draws N fresh segments from the pool and attaches them to the item under the
    `ICWM_CONTEXT_*` observation keys consumed by `policies/icwm/processor_icwm.py`.

Note on scope: this file is training-data plumbing only. It is intentionally not wired into
an actual training run in this deliverable (see icwm/README.md: no GPU is available on this
machine right now), so it has not been exercised end-to-end against a full-size dataset --
only against the parquet-level schema of a real rendered LIBERO-Spatial LeRobot dataset
(`/workspace/yujin/data/libero_spatial_rot/lerobot_v30/...`) to make sure the column names
referenced below (`dataset_from_index`, `dataset_to_index`, `length`) are correct for the
v3.0 LeRobot dataset format actually present on this machine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.icwm.processor_icwm import (
    ICWM_CONTEXT_ACTIONS,
    ICWM_CONTEXT_N_VALID,
    ICWM_CONTEXT_OBS_END,
    ICWM_CONTEXT_OBS_START,
    ICWM_HISTORY_ACTIONS,
    ICWM_HISTORY_N_VALID,
    ICWM_HISTORY_OBS_END,
    ICWM_HISTORY_OBS_START,
)

logger = logging.getLogger(__name__)


@dataclass
class ICWMInteractionSegmentPool:
    """A pool of (dataset, start_index) interaction segments drawn from many trajectories/viewpoints.

    Each segment is a candidate (o^s, a, o^e) triple: `horizon` consecutive actions starting
    at `start_index`, with `o^s` the observation at `start_index` and `o^e` the observation
    `horizon` steps later. Segments never cross an episode boundary.

    Building the *full* set of valid (dataset, start_index) pairs across every training
    trajectory can be large (O(total_frames)); `stride` subsamples candidate start indices
    within each episode to bound pool size/memory (the paper does not specify a particular
    pool granularity -- any sufficiently diverse/large pool satisfies "sampled from a pool of
    interaction segments collected across all training trajectories and viewpoints").
    """

    datasets: list[LeRobotDataset]
    horizon: int
    stride: int = 1
    image_key: str = "observation.images.image"
    action_key: str = "action"

    _segments: list[tuple[int, int]] = field(default_factory=list, init=False, repr=False)
    _segment_episode: list[int] = field(default_factory=list, init=False, repr=False)
    _fast: bool = field(default=False, init=False, repr=False)
    _actions: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if not self.datasets:
            raise ValueError("ICWMInteractionSegmentPool requires at least one dataset.")

        for d_idx, ds in enumerate(self.datasets):
            episodes = ds.meta.episodes  # datasets.Dataset with one row per episode
            for row in episodes:
                start = int(row["dataset_from_index"])
                end = int(row["dataset_to_index"])  # exclusive
                last_valid_start = end - self.horizon
                for s in range(start, max(start, last_valid_start), self.stride):
                    self._segments.append((d_idx, s))
                    self._segment_episode.append(int(row["episode_index"]))

        from lerobot.datasets import icwm_frame_cache

        names = [Path(ds.root).name for ds in self.datasets]
        if icwm_frame_cache.install() and all(icwm_frame_cache.available(n) for n in names):
            self._fast = True
            self._actions = [
                np.asarray(ds.hf_dataset.select_columns(self.action_key)[self.action_key], dtype=np.float32)
                for ds in self.datasets
            ]
            self._names = names
            logger.info("ICWMInteractionSegmentPool: using the frame cache for %s", names)

        if not self._segments:
            raise ValueError(
                f"No valid interaction segments found (horizon={self.horizon} may be larger "
                "than every episode in the provided datasets)."
            )
        logger.info(
            "ICWMInteractionSegmentPool: %d candidate segments from %d dataset(s) (horizon=%d, stride=%d)",
            len(self._segments),
            len(self.datasets),
            self.horizon,
            self.stride,
        )

    def __len__(self) -> int:
        return len(self._segments)

    def sample(
        self, n: int, rng: np.random.Generator, exclude_episode: int | None = None
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Sample N (o^s, a, o^e) interaction clips, independently and with replacement.

        Returns a list of `n` tuples `(o_s, a, o_e)` where `o_s`/`o_e` are single images
        (C, H, W) and `a` is the (horizon, action_dim) action chunk executed between them.
        """
        chosen = rng.integers(0, len(self._segments), size=n)
        if exclude_episode is not None:
            # never hand the query its own episode: redraw any clip that came from it
            seg_ep = np.asarray(self._segment_episode)
            for _ in range(100):
                bad = seg_ep[chosen] == exclude_episode
                if not bad.any():
                    break
                chosen[bad] = rng.integers(0, len(self._segments), size=int(bad.sum()))
            else:
                raise RuntimeError("could not draw clips outside the query's episode")
        return [self.clip_at(*self._segments[i]) for i in chosen]

    def clip_at(self, d_idx: int, start: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The (o^s, o^e, a) clip of `datasets[d_idx]` starting at global frame index `start`."""
        if self._fast:
            from lerobot.datasets import icwm_frame_cache

            mm = icwm_frame_cache.frames(self._names[d_idx], self.image_key)
            chw = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div_(255.0)  # noqa: E731
            acts = torch.from_numpy(self._actions[d_idx][start : start + self.horizon].copy())
            return chw(mm[start]), chw(mm[start + self.horizon]), acts
        ds = self.datasets[d_idx]
        start_item = ds[start]
        end_item = ds[start + self.horizon]
        # NOTE: fetching each of the `horizon` action vectors with a separate __getitem__
        # call is simple/correct but not efficient; a production version should instead
        # configure `ds` with `delta_timestamps={"action": [...]}` so a single __getitem__
        # returns the whole action chunk directly (see how policies/*/configuration_*.py
        # `action_delta_indices` is normally consumed by `lerobot.scripts.lerobot_train`).
        actions = torch.stack([ds[start + t][self.action_key] for t in range(self.horizon)])
        return start_item[self.image_key], end_item[self.image_key], actions

    def blank_clip(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """No information: black frames and a motionless arm with the gripper open (= eval `blank_clips`)."""
        black = torch.zeros_like(like)
        acts = torch.zeros(self.horizon, self.datasets[0].meta.features[self.action_key]["shape"][0])
        acts[:, -1] = -1.0  # LIBERO: last action dim is the gripper
        return black, black, acts

    def history(self, d_idx: int, query: int, ep_start: int, n: int, max_gap: int,
                rng: np.random.Generator) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """The n clips of the query's own episode that precede global frame `query`, oldest first.

        Clip k starts horizon + g_k frames before clip k+1 (clip n: before the query), g_k ~ U[0, max_gap],
        so its end frame is at or before the next clip's start. Clips that would start before
        `ep_start` do not exist; they are returned as leading blank placeholders (so the batch can
        be stacked) and the second return value is the number of real clips.
        """
        starts, s = [], query
        for _ in range(n):
            s = s - self.horizon - int(rng.integers(0, max_gap + 1))
            starts.append(s)
        starts.reverse()
        real = [self.clip_at(d_idx, s) for s in starts if s >= ep_start]
        like = real[0][0] if real else self.clip_at(d_idx, ep_start)[0]
        return [self.blank_clip(like)] * (n - len(real)) + real, len(real)

    def same_task_past(self, d_idx: int, episode: int, t: int, n: int,
                       rng: np.random.Generator) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """n clips from OTHER episodes of the query's task, all ending at or before episode step `t`.

        Each clip: a random other episode of the same task, a random start offset s with
        s + horizon <= min(t, len(episode)), ordered by s (oldest first). If t < horizon no clip fits:
        n blank placeholders and 0 real clips (same convention as `history`).
        """
        ds = self.datasets[d_idx]
        if not hasattr(self, "_task_eps"):
            self._task_eps = {}
        if d_idx not in self._task_eps:
            task_col = np.asarray(ds.hf_dataset.select_columns("task_index")["task_index"]).reshape(-1)
            eps, by_task = {}, {}
            for row in ds.meta.episodes:
                a, b = int(row["dataset_from_index"]), int(row["dataset_to_index"])
                e = int(row["episode_index"])
                eps[e] = (a, b, int(task_col[a]))
                by_task.setdefault(int(task_col[a]), []).append(e)
            self._task_eps[d_idx] = (eps, by_task)
        eps, by_task = self._task_eps[d_idx]
        a, _, task = eps[episode]
        others = [e for e in by_task[task] if e != episode]
        picks = []
        for _ in range(n):
            e = others[int(rng.integers(0, len(others)))]
            ea, eb, _ = eps[e]
            last = min(t, eb - ea - 1) - self.horizon  # end frame s+horizon must stay inside episode e
            if last >= 0:
                picks.append((int(rng.integers(0, last + 1)), ea))
        picks.sort()
        real = [self.clip_at(d_idx, ea + s) for s, ea in picks]
        like = real[0][0] if real else self.clip_at(d_idx, a)[0]
        return [self.blank_clip(like)] * (n - len(real)) + real, len(real)


class ICWMContextSampledDataset(Dataset):
    """Wraps a task dataset, attaching N freshly-sampled interaction clips to every item.

    `task_dataset` supplies the real (o_t, l, a_t) training query (any `LeRobotDataset`,
    typically the union, across the 8 training azimuths, of episodes for one LIBERO suite).
    `context_pool` supplies the N task-agnostic interaction clips T prepended to it.

    The two need not be the same dataset(s): per the paper, T is drawn from a pool spanning
    *all* training trajectories/viewpoints, while the query can be restricted to a particular
    suite/split.
    """

    def __init__(
        self,
        task_dataset: LeRobotDataset,
        context_pool: ICWMInteractionSegmentPool,
        n_context_clips: int = 5,
        seed: int = 0,
    ):
        self.task_dataset = task_dataset
        self.context_pool = context_pool
        self.n_context_clips = n_context_clips
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.task_dataset)

    @property
    def meta(self):
        """Proxy to the wrapped task dataset's metadata.

        `lerobot.scripts.lerobot_train` accesses `dataset.meta.*` (stats, episodes,
        camera_keys, has_language_columns, ...) directly; since this class is a thin
        `torch.utils.data.Dataset` wrapper (not a `LeRobotDataset` subclass), it must forward
        those lookups to the underlying task dataset it wraps.
        """
        return self.task_dataset.meta

    # `lerobot.scripts.lerobot_train` also reaches past `.meta` for a few convenience
    # properties/attributes directly on `dataset` itself (num_frames/num_episodes for
    # logging, hf_dataset/episodes for the episode-aware sampler) -- forward those too.
    @property
    def num_frames(self):
        return self.task_dataset.num_frames

    @property
    def num_episodes(self):
        return self.task_dataset.num_episodes

    @property
    def hf_dataset(self):
        return self.task_dataset.hf_dataset

    @property
    def episodes(self):
        return self.task_dataset.episodes

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.task_dataset[idx])
        if self.n_context_clips == 0:
            return item

        clips = self.context_pool.sample(self.n_context_clips, self._rng)
        obs_start = torch.stack([c[0] for c in clips])  # (N, C, H, W)
        obs_end = torch.stack([c[1] for c in clips])  # (N, C, H, W)
        actions = torch.stack([c[2] for c in clips])  # (N, horizon, action_dim)

        item[ICWM_CONTEXT_OBS_START] = obs_start
        item[ICWM_CONTEXT_OBS_END] = obs_end
        item[ICWM_CONTEXT_ACTIONS] = actions
        return item


def build_icwm_context_pool_from_azimuth_datasets(
    dataset_roots: list[str],
    horizon: int,
    stride: int = 20,
    image_key: str = "observation.images.image",
    action_key: str = "action",
) -> ICWMInteractionSegmentPool:
    """Convenience constructor: load one `LeRobotDataset` per rendered-azimuth root directory
    and build a single pooled `ICWMInteractionSegmentPool` across all of them.

    Example (Table 3 protocol, 8 in-domain LIBERO-Spatial azimuths):

        pool = build_icwm_context_pool_from_azimuth_datasets(
            dataset_roots=[
                f"/workspace/yujin/data/libero_spatial_rot/lerobot_v30/"
                f"libero_spatial_theta{theta:03d}_no_noops_lerobot"
                for theta in [30, 60, 90, 120, 240, 270, 300, 330]
            ],
            horizon=5,  # action_chunk_size
        )
    """
    datasets = [LeRobotDataset(repo_id=root, root=root) for root in dataset_roots]
    return ICWMInteractionSegmentPool(
        datasets=datasets,
        horizon=horizon,
        stride=stride,
        image_key=image_key,
        action_key=action_key,
    )


class ICWMViewpointMatchedDataset(Dataset):
    """Query = union of several viewpoint datasets; context clips come from the query's own viewpoint.

    Each root is one system configuration psi (one camera azimuth). For a query drawn from
    dataset k, the N interaction clips are sampled from dataset k's own segment pool -- any task,
    any episode, but the same viewpoint -- so T genuinely carries information about the psi the
    query was recorded under. (The pooled-across-viewpoints variant in `ICWMContextSampledDataset`
    attaches a clip from a different viewpoint to most queries, which makes T uninformative about
    psi and lets the model learn to ignore it.)

    `query_datasets` supply (o_t, l, a_t) and must be built the way `make_dataset` builds the
    task dataset (delta timestamps, uint8 images); `pool_datasets` are the matching roots loaded
    for clip sampling. Normalisation statistics are aggregated over all query datasets.
    """

    def __init__(self, query_datasets, pool_datasets, horizon, n_context_clips=5, stride=20,
                 image_key="observation.images.image", action_key="action", seed=0,
                 context_source="pool", history_max_gap=15, probe_dir=None,
                 n_history_clips=5, drop_history_p=0.0, drop_explore_p=0.0):
        from lerobot.datasets.compute_stats import aggregate_stats

        assert len(query_datasets) == len(pool_datasets)
        assert context_source in ("pool", "history", "sametask_past", "probe", "mixed"), context_source
        self.context_source = context_source
        self.n_history_clips = n_history_clips
        self.drop_history_p = drop_history_p
        self.drop_explore_p = drop_explore_p
        self.probe_dir = probe_dir
        self._probes = {}
        self.history_max_gap = history_max_gap
        self.query_datasets = query_datasets
        self.pools = [
            ICWMInteractionSegmentPool(datasets=[d], horizon=horizon, stride=stride,
                                       image_key=image_key, action_key=action_key)
            for d in pool_datasets
        ]
        self.n_context_clips = n_context_clips
        self._rng = np.random.default_rng(seed)
        self._cum = np.cumsum([0] + [len(d) for d in query_datasets])
        self._meta = query_datasets[0].meta
        self._meta.stats = aggregate_stats([d.meta.stats for d in query_datasets])

    def __len__(self) -> int:
        return int(self._cum[-1])

    @property
    def meta(self):
        return self._meta

    @property
    def num_frames(self):
        return sum(d.num_frames for d in self.query_datasets)

    @property
    def num_episodes(self):
        return sum(d.num_episodes for d in self.query_datasets)

    @property
    def hf_dataset(self):
        return self.query_datasets[0].hf_dataset

    @property
    def episodes(self):
        return self.query_datasets[0].episodes

    def _probe_clips(self, k: int, task_index: int):
        """n_context_clips probing clips (uniform, with replacement) of scene (dataset k, task_index)."""
        key = (k, task_index)
        if key not in self._probes:  # memmaps opened lazily, once per DataLoader worker
            from pathlib import Path

            tag = Path(self.probe_dir) / f"{Path(str(self.query_datasets[k].root)).name}__task{task_index}"
            if not Path(f"{tag}__meta.json").exists():
                raise FileNotFoundError(f"no finished probe pool {tag}__*")
            self._probes[key] = (np.load(f"{tag}__start.npy", mmap_mode="r"),
                                 np.load(f"{tag}__end.npy", mmap_mode="r"),
                                 np.load(f"{tag}__actions.npy"))
        starts, ends, acts = self._probes[key]
        chw = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div_(255.0)  # noqa: E731
        # fresh entropy per call: self._rng is copied identically into every forked DataLoader worker
        pick = np.random.default_rng().integers(0, len(acts), size=self.n_context_clips)
        return [(chw(starts[i]), chw(ends[i]), torch.from_numpy(acts[i].copy())) for i in pick]

    def __getitem__(self, idx: int) -> dict:
        k = int(np.searchsorted(self._cum, idx, side="right") - 1)
        item = dict(self.query_datasets[k][int(idx - self._cum[k])])
        if self.n_context_clips == 0:
            return item
        if self.context_source == "mixed":
            # exploration segment ("pool": same viewpoint, other episodes) + history segment (own episode)
            ep = self.query_datasets[k].meta.episodes[int(item["episode_index"])]
            hist, n_hist = self.pools[k].history(0, int(item["index"]), int(ep["dataset_from_index"]),
                                                 self.n_history_clips, self.history_max_gap, self._rng)
            clips = self.pools[k].sample(self.n_context_clips, self._rng, exclude_episode=int(item["episode_index"]))
            n_explore = self.n_context_clips
            # context dropout: keep one segment only, so each source stays usable on its own.
            # fresh entropy: self._rng is copied identically into every forked DataLoader worker
            u = np.random.default_rng().random()
            if u < self.drop_history_p:
                n_hist = 0
            elif u < self.drop_history_p + self.drop_explore_p:
                n_explore = 0
            item[ICWM_CONTEXT_N_VALID] = torch.tensor(n_explore)
            item[ICWM_HISTORY_N_VALID] = torch.tensor(n_hist)
            item[ICWM_HISTORY_OBS_START] = torch.stack([c[0] for c in hist])
            item[ICWM_HISTORY_OBS_END] = torch.stack([c[1] for c in hist])
            item[ICWM_HISTORY_ACTIONS] = torch.stack([c[2] for c in hist])
        elif self.context_source == "history":
            # the query's own episode: the chunks executed just before it (Reflective-VLA style)
            ep = self.query_datasets[k].meta.episodes[int(item["episode_index"])]
            clips, n_real = self.pools[k].history(0, int(item["index"]), int(ep["dataset_from_index"]),
                                                  self.n_context_clips, self.history_max_gap, self._rng)
            # Reflective VLA masks frames before the episode start: only the real clips enter the sequence
            item[ICWM_CONTEXT_N_VALID] = torch.tensor(n_real)
        elif self.context_source == "sametask_past":
            # other demos of the query's task, only segments that end before the query's step
            ep = self.query_datasets[k].meta.episodes[int(item["episode_index"])]
            t = int(item["index"]) - int(ep["dataset_from_index"])
            clips, n_real = self.pools[k].same_task_past(0, int(item["episode_index"]), t,
                                                         self.n_context_clips, self._rng)
            item[ICWM_CONTEXT_N_VALID] = torch.tensor(n_real)
        elif self.context_source == "probe":
            # self-exploration clips of the query's own scene, as recorded at test time
            clips = self._probe_clips(k, int(item["task_index"]))
        else:
            # clips from the query's own viewpoint, but never from the query's own episode
            clips = self.pools[k].sample(self.n_context_clips, self._rng, exclude_episode=int(item["episode_index"]))
        item[ICWM_CONTEXT_OBS_START] = torch.stack([c[0] for c in clips])
        item[ICWM_CONTEXT_OBS_END] = torch.stack([c[1] for c in clips])
        item[ICWM_CONTEXT_ACTIONS] = torch.stack([c[2] for c in clips])
        return item
