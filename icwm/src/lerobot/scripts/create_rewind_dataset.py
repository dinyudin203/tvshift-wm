#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Create a time-reversed ("ReWiND") copy of a LeRobot (v3.0) dataset.

For every episode the observation frames are played back in reverse, and a matching
reverse-action label is synthesized. This augments a goal-conditioned IDM with diverse
``(o_t, o_{t+H}, action)`` tuples (the same o_t is experienced moving toward several
different goals), which is hypothesized to make goal following more robust.

Action reversal (assumes the standard LeRobot/LIBERO convention ``action[t]`` is applied
at ``o_t`` and produces ``o_{t+1}``, i.e. conv1, verified empirically on LIBERO):

    Reversed trajectory visits õ_i = o_{N-1-i}. The reverse step õ_i -> õ_{i+1}
    (= o_{N-1-i} -> o_{N-2-i}) is the inverse of the forward step that produced
    o_{N-1-i} from o_{N-2-i}, whose action is a_{N-2-i}. Therefore:

        rewound_action[i] = -a_{N-2-i}          (end-effector / delta dims)
        rewound_action[i] =  a_{N-2-i}          (gripper / absolute dims, NOT negated)

    for i in [0, L-2]; the final reversed frame õ_{L-1} = o_0 is terminal and its action
    is copy-padded from the previous frame (mirroring the trailing action in the forward
    data).

Two subtleties this script gets right (vs. a naive ``-take(action)``):
  1. **Off-by-one**: the reverse action at õ_i must come from forward index ``N-2-i``
     (the action that *arrived* at frame ``N-1-i``), not ``N-1-i`` (the action that
     *left* it). Negating ``a_{N-1-i}`` mislabels each step by one frame.
  2. **Gripper is absolute, not a delta**: the gripper command is a binary/absolute
     open-close target. Negating it flips the whole open/close signal, which is wrong.
     The reversed gripper is the time-reversed *value*, never negated.

Implementation: each data file holds whole episodes (episodes never span files), and
images are stored inline in the parquet rows (``struct<bytes, path>``). So we process one
data file at a time -- reversing the row order *within* each episode (which reverses the
observation frames, including the inline images) and recomputing the action column,
indices and timestamps. The original file partitioning, episode boundaries and global
``index`` sequence are preserved, so only the per-episode/global action stats need to be
updated in the metadata. Per-file processing also avoids pyarrow's 2 GiB single-array
offset limit that a whole-dataset concatenation of image bytes would hit.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ACTION_STAT_QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def _dataset_dirs(root: Path) -> list[Path]:
    if (root / "meta" / "info.json").exists():
        return [root]
    dirs = sorted(p for p in root.iterdir() if (p / "meta" / "info.json").exists())
    if not dirs:
        raise FileNotFoundError(f"No LeRobot dataset directories found under {root}")
    return dirs


def _data_files(dataset_dir: Path) -> list[Path]:
    files = sorted((dataset_dir / "data").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir / 'data'}")
    return files


def _replace_column(table: pa.Table, name: str, values, type_: pa.DataType | None = None) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        return table
    array = pa.array(values, type=type_ or table.schema.field(name).type)
    return table.set_column(idx, table.schema.field(name), array)


def _action_stats(values: np.ndarray) -> dict:
    stats = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }
    for key, q in ACTION_STAT_QUANTILES.items():
        stats[key] = np.quantile(values, q, axis=0).tolist()
    return stats


def _reverse_episode_actions(forward: np.ndarray, gripper_dims: tuple[int, ...]) -> np.ndarray:
    """Synthesize the reverse-action labels for one episode (conv1).

    Args:
        forward: (L, D) forward action array for the episode.
        gripper_dims: indices of absolute (gripper) dims that must NOT be negated.

    Returns:
        (L, D) reverse-action array.
    """
    length, dim = forward.shape
    rev = np.empty_like(forward)
    if length == 1:
        # Degenerate episode: nothing to reverse.
        return forward.copy()

    # src[i] = forward[L-2-i] for i in [0, L-2]  (i.e. reverse of forward[:-1]).
    src = forward[length - 2 :: -1]  # shape (L-1, D)
    delta_mask = np.ones(dim, dtype=bool)
    for g in gripper_dims:
        delta_mask[g] = False
    rev[: length - 1, delta_mask] = -src[:, delta_mask]  # delta/eef dims: negate
    rev[: length - 1, ~delta_mask] = src[:, ~delta_mask]  # absolute/gripper dims: keep
    rev[length - 1] = rev[length - 2]  # terminal frame: copy-pad
    return rev


def _episode_blocks(episode_index: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous (start, end) local row ranges, one per episode, in order of appearance."""
    blocks: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(episode_index) + 1):
        if i == len(episode_index) or episode_index[i] != episode_index[start]:
            blocks.append((start, i))
            start = i
    return blocks


def _rewind_data_file(
    in_path: Path,
    out_path: Path,
    fps: int,
    gripper_dims: tuple[int, ...],
    rev_actions_by_episode: dict[int, np.ndarray],
) -> None:
    """Reverse rows within each episode of a single data file and rebuild action/indices.

    Preserves the file's global ``index`` sequence and episode boundaries; only frame
    order (incl. inline images), the action column, ``frame_index`` and ``timestamp`` are
    changed. Collects per-episode reverse actions into ``rev_actions_by_episode`` for stats.
    """
    table = pq.read_table(in_path)
    episode_index = np.asarray(table.column("episode_index").to_pylist())
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)

    order: list[int] = []
    frame_indices: list[int] = []
    timestamps: list[float] = []
    rev_blocks: list[np.ndarray] = []
    for start, end in _episode_blocks(episode_index):
        length = end - start
        order.extend(range(end - 1, start - 1, -1))  # reverse rows within the episode
        rev = _reverse_episode_actions(actions[start:end], gripper_dims)
        rev_blocks.append(rev)
        rev_actions_by_episode[int(episode_index[start])] = rev
        frame_indices.extend(range(length))
        timestamps.extend(i / fps for i in range(length))

    rewound = table.take(pa.array(order, type=pa.int64()))
    action_type = rewound.schema.field("action").type
    rev_actions = np.concatenate(rev_blocks, axis=0).astype(np.float32).tolist()

    rewound = _replace_column(rewound, "action", rev_actions, action_type)
    rewound = _replace_column(rewound, "frame_index", frame_indices, pa.int64())
    rewound = _replace_column(rewound, "timestamp", timestamps, pa.float32())
    # Keep the original ascending global ``index`` (row positions are unchanged); episode
    # and task indices are constant within each (reversed) episode block, so they survive
    # the ``take`` untouched.
    rewound = _replace_column(rewound, "index", table.column("index").to_pylist(), pa.int64())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(rewound, out_path)


def _update_episode_action_stats(episodes_df, rev_actions_by_episode: dict[int, np.ndarray]):
    out = episodes_df.copy()
    for i, row in out.iterrows():
        eid = int(row["episode_index"])
        if eid not in rev_actions_by_episode:
            continue
        stats = _action_stats(rev_actions_by_episode[eid])
        for stat_key, stat_value in stats.items():
            col = f"stats/action/{stat_key}"
            if col in out.columns:
                out.at[i, col] = stat_value
    return out


def _rewind_one_dataset(
    input_dir: Path,
    output_dir: Path,
    gripper_dims: tuple[int, ...],
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    # Copy all metadata first; we then overwrite the action stats in-place.
    shutil.copytree(input_dir / "meta", output_dir / "meta", dirs_exist_ok=True)

    with open(input_dir / "meta" / "info.json") as f:
        info = json.load(f)
    fps = int(info["fps"])

    rev_actions_by_episode: dict[int, np.ndarray] = {}
    for in_path in _data_files(input_dir):
        rel = in_path.relative_to(input_dir)
        _rewind_data_file(in_path, output_dir / rel, fps, gripper_dims, rev_actions_by_episode)

    all_actions = np.concatenate(list(rev_actions_by_episode.values()), axis=0)

    # Update global stats.json (action only).
    stats_path = output_dir / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        if "action" in stats:
            stats["action"] = _action_stats(all_actions)
            with open(stats_path, "w") as f:
                json.dump(stats, f)

    # Update per-episode action stats in meta/episodes/*.
    ep_files = sorted((output_dir / "meta" / "episodes").glob("*/*.parquet"))
    for ep_path in ep_files:
        et = pq.read_table(ep_path)
        edf = et.to_pandas()
        edf = _update_episode_action_stats(edf, rev_actions_by_episode)
        pq.write_table(pa.Table.from_pandas(edf, schema=et.schema, preserve_index=False), ep_path)


def create_rewind_dataset(
    input_root: Path,
    output_root: Path,
    gripper_dims: tuple[int, ...],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for input_dir in _dataset_dirs(input_root):
        output_dir = output_root / input_dir.name if input_dir != input_root else output_root
        print(f"rewind {input_dir} -> {output_dir} (gripper_dims={gripper_dims})")
        _rewind_one_dataset(input_dir, output_dir, gripper_dims=gripper_dims)


def _combined_rev_file(
    in_path: Path,
    out_path: Path,
    fps: int,
    gripper_dims: tuple[int, ...],
    ep_offset: int,
    frame_offset: int,
    rev_actions_by_episode: dict[int, np.ndarray],
) -> None:
    """Like ``_rewind_data_file`` but SHIFTS episode_index by ``ep_offset`` and the global
    ``index`` by ``frame_offset`` so the reversed episodes can be appended *after* the
    originals in a combined dataset. Reverse actions are keyed by the NEW episode index."""
    table = pq.read_table(in_path)
    episode_index = np.asarray(table.column("episode_index").to_pylist())
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)

    order: list[int] = []
    frame_indices: list[int] = []
    timestamps: list[float] = []
    rev_blocks: list[np.ndarray] = []
    for start, end in _episode_blocks(episode_index):
        length = end - start
        order.extend(range(end - 1, start - 1, -1))
        rev = _reverse_episode_actions(actions[start:end], gripper_dims)
        rev_blocks.append(rev)
        rev_actions_by_episode[int(episode_index[start]) + ep_offset] = rev
        frame_indices.extend(range(length))
        timestamps.extend(i / fps for i in range(length))

    rewound = table.take(pa.array(order, type=pa.int64()))
    action_type = rewound.schema.field("action").type
    rev_actions = np.concatenate(rev_blocks, axis=0).astype(np.float32).tolist()
    rewound = _replace_column(rewound, "action", rev_actions, action_type)
    rewound = _replace_column(rewound, "frame_index", frame_indices, pa.int64())
    rewound = _replace_column(rewound, "timestamp", timestamps, pa.float32())
    rewound = _replace_column(rewound, "episode_index", (episode_index + ep_offset).tolist(), pa.int64())
    new_index = (np.asarray(table.column("index").to_pylist()) + frame_offset).tolist()
    rewound = _replace_column(rewound, "index", new_index, pa.int64())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(rewound, out_path)


def create_combined_dataset(
    input_root: Path,
    output_root: Path,
    gripper_dims: tuple[int, ...],
) -> None:
    """Write ORIGINAL + time-REVERSED as one dataset (2x episodes).

    The original ``N`` episodes are kept verbatim; the reversed copies are appended as
    episodes ``N..2N-1`` in new data files (file_index += n_files), with the global frame
    ``index`` shifted by ``total_frames``. This is the *intended* ReWiND training set: the
    same o_t appears moving toward both a forward and a reverse goal.
    """
    import pandas as pd

    dirs = _dataset_dirs(input_root)
    if len(dirs) != 1:
        raise ValueError(f"combined mode expects a single dataset dir, got {len(dirs)}: {dirs}")
    input_dir = dirs[0]

    if output_root.exists():
        shutil.rmtree(output_root)
    print(f"combined: copy original {input_dir} -> {output_root}")
    shutil.copytree(input_dir, output_root)

    with open(input_dir / "meta" / "info.json") as f:
        info = json.load(f)
    fps = int(info["fps"])
    n_ep = int(info["total_episodes"])
    n_frames = int(info["total_frames"])
    chunks_size = int(info["chunks_size"])
    data_path_tmpl = info["data_path"]
    data_files = _data_files(input_dir)
    n_files = len(data_files)

    # Reversed data files appended after the originals.
    rev_actions_by_episode: dict[int, np.ndarray] = {}
    for k, in_path in enumerate(data_files):
        new_fi = n_files + k
        new_ci = new_fi // chunks_size
        out_rel = data_path_tmpl.format(chunk_index=new_ci, file_index=new_fi)
        print(f"  reverse {in_path.name} -> {out_rel}")
        _combined_rev_file(
            in_path, output_root / out_rel, fps, gripper_dims, n_ep, n_frames, rev_actions_by_episode
        )

    # meta/episodes: append reversed episode rows (episode_index += N, file_index += n_files,
    # dataset_from/to_index += total_frames, action stats recomputed).
    ep_files = sorted((input_dir / "meta" / "episodes").glob("*/*.parquet"))
    if len(ep_files) != 1:
        raise NotImplementedError(f"combined mode assumes a single episodes file, got {len(ep_files)}")
    et = pq.read_table(ep_files[0])
    edf = et.to_pandas()
    rev = edf.copy()
    rev["episode_index"] = rev["episode_index"] + n_ep
    rev["data/file_index"] = rev["data/file_index"] + n_files
    rev["dataset_from_index"] = rev["dataset_from_index"] + n_frames
    rev["dataset_to_index"] = rev["dataset_to_index"] + n_frames
    rev = _update_episode_action_stats(rev, rev_actions_by_episode)
    combined = pd.concat([edf, rev], ignore_index=True)
    out_ep = output_root / "meta" / "episodes" / ep_files[0].relative_to(input_dir / "meta" / "episodes")
    pq.write_table(pa.Table.from_pandas(combined, schema=et.schema, preserve_index=False), out_ep)

    # info.json: double the episode/frame counts and the train split.
    info["total_episodes"] = 2 * n_ep
    info["total_frames"] = 2 * n_frames
    info["splits"] = {"train": f"0:{2 * n_ep}"}
    with open(output_root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # stats.json: recompute global ACTION stats over original + reversed (image/state are the
    # same value set under reversal, so their stats are unchanged).
    stats_path = output_root / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        if "action" in stats:
            orig_actions = np.concatenate(
                [np.asarray(pq.read_table(p, columns=["action"]).column("action").to_pylist(),
                            dtype=np.float32) for p in data_files],
                axis=0,
            )
            rev_all = np.concatenate(list(rev_actions_by_episode.values()), axis=0)
            stats["action"] = _action_stats(np.concatenate([orig_actions, rev_all], axis=0))
            with open(stats_path, "w") as f:
                json.dump(stats, f)
    print(f"combined dataset written: {2 * n_ep} episodes, {2 * n_frames} frames -> {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a time-reversed (ReWiND) copy of a LeRobot dataset.")
    parser.add_argument(
        "--input-root", type=Path, required=True, help="Path to a LeRobot dataset dir (or a dir of them)."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["reversed", "combined"],
        default="reversed",
        help="reversed: replace each episode by its time-reversed copy (same #episodes). "
        "combined: ORIGINAL + reversed appended (2x episodes) — the intended ReWiND training set.",
    )
    parser.add_argument(
        "--gripper-dims",
        type=int,
        nargs="*",
        default=[-1],
        help="Action dims that are absolute (gripper) and must NOT be negated when reversing. "
        "Default: the last dim (LIBERO: [dx,dy,dz,droll,dpitch,dyaw, gripper]). "
        "Pass an empty list to negate all dims (legacy behavior).",
    )
    args = parser.parse_args()

    # Resolve negative gripper indices against the action dimension.
    dirs = _dataset_dirs(args.input_root)
    sample = pq.read_table(_data_files(dirs[0])[0], columns=["action"]).column("action")[0].as_py()
    action_dim = len(sample)
    gripper_dims = tuple(d % action_dim for d in args.gripper_dims)

    if args.mode == "combined":
        create_combined_dataset(args.input_root, args.output_root, gripper_dims=gripper_dims)
    else:
        create_rewind_dataset(args.input_root, args.output_root, gripper_dims=gripper_dims)
    print(args.output_root)


if __name__ == "__main__":
    main()
