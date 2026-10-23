#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
from pprint import pformat

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_PREFIX, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .streaming_dataset import StreamingLeRobotDataset


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    dataset = _maybe_wrap_icwm_context_sampler(cfg, dataset)

    return dataset


def _maybe_wrap_icwm_context_sampler(cfg: TrainPipelineConfig, dataset):
    """Wrap `dataset` with `ICWMContextSampledDataset` when training an ICWM policy.

    ICWM (arXiv 2606.26025) supervises `pi_theta(a_t | Psi(T), o_t, l)`, where T is a set of
    `n_context_clips` interaction clips randomly sampled from a pool spanning all training
    trajectories/viewpoints (paper Sec 4.1) -- not from the same episode as the (o_t, l, a_t)
    query. This is deliberately a no-op unless the ICWM policy config both requests context
    clips (`n_context_clips > 0`) and supplies `context_pool_dataset_roots` (e.g. the 8
    in-domain training-azimuth dataset roots for one LIBERO suite); otherwise the ICWM
    processor gracefully degrades to the `n_context_clips=0` (MV baseline) behavior.
    """
    from lerobot.policies.icwm.configuration_icwm import ICWMConfig

    policy_cfg = cfg.trainable_config
    if not isinstance(policy_cfg, ICWMConfig) or policy_cfg.n_context_clips <= 0:
        return dataset
    if not policy_cfg.context_pool_dataset_roots:
        logging.warning(
            "ICWMConfig.n_context_clips=%d but context_pool_dataset_roots is empty: training "
            "will proceed WITHOUT interaction-context clips (equivalent to the MV baseline, "
            "n_context_clips=0), not full ICWM. Set --policy.context_pool_dataset_roots=[...] "
            "to enable real ICWM training.",
            policy_cfg.n_context_clips,
        )
        return dataset

    from lerobot.datasets.icwm_context_sampler import (
        ICWMContextSampledDataset,
        ICWMViewpointMatchedDataset,
        build_icwm_context_pool_from_azimuth_datasets,
    )

    from lerobot.datasets import icwm_frame_cache

    if icwm_frame_cache.install():  # main process, before the DataLoader workers fork
        logging.info("ICWM frame cache active: %s", icwm_frame_cache.cache_dir())

    if getattr(policy_cfg, "context_match_viewpoint", False):
        # Queries come from the union of all roots and each query gets clips from its own viewpoint.
        roots = list(policy_cfg.context_pool_dataset_roots)
        image_transforms = (
            ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
        )
        query_datasets = []
        for root in roots:
            meta = LeRobotDatasetMetadata(root, root=root)
            query_datasets.append(
                LeRobotDataset(
                    root,
                    root=root,
                    delta_timestamps=resolve_delta_timestamps(cfg.trainable_config, meta),
                    image_transforms=image_transforms,
                    video_backend=cfg.dataset.video_backend,
                    return_uint8=True,
                    tolerance_s=cfg.tolerance_s,
                )
            )
        pool_datasets = [LeRobotDataset(root, root=root) for root in roots]
        logging.info("ICWM viewpoint-matched training: %d viewpoint datasets, %d query frames",
                     len(roots), sum(len(d) for d in query_datasets))
        return ICWMViewpointMatchedDataset(
            query_datasets=query_datasets,
            pool_datasets=pool_datasets,
            horizon=policy_cfg.action_chunk_size,
            n_context_clips=policy_cfg.n_context_clips,
            stride=policy_cfg.context_pool_stride,
            context_source=getattr(policy_cfg, "context_source", "pool"),
            history_max_gap=getattr(policy_cfg, "context_history_max_gap", 15),
            probe_dir=getattr(policy_cfg, "context_probe_dir", None),
            n_history_clips=getattr(policy_cfg, "context_mixed_n_history", 5),
            drop_history_p=getattr(policy_cfg, "context_drop_history_p", 0.0),
            drop_explore_p=getattr(policy_cfg, "context_drop_explore_p", 0.0),
        )

    logging.info(
        "Building ICWM interaction-clip context pool from %d dataset root(s): %s",
        len(policy_cfg.context_pool_dataset_roots),
        pformat(policy_cfg.context_pool_dataset_roots),
    )
    pool = build_icwm_context_pool_from_azimuth_datasets(
        dataset_roots=policy_cfg.context_pool_dataset_roots,
        horizon=policy_cfg.action_chunk_size,
        stride=policy_cfg.context_pool_stride,
    )
    return ICWMContextSampledDataset(
        task_dataset=dataset,
        context_pool=pool,
        n_context_clips=policy_cfg.n_context_clips,
    )
