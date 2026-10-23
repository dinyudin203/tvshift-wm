# Copyright 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pre/post-processing pipelines for Reflective VLA.

Two differences from pi0.5's pipeline:

1. Proprioception leaves the prompt.  pi0.5 discretizes the state into 256 bins and splices
   it into the instruction string; the paper instead projects "proprioceptive states and
   historical action chunks into the token space by lightweight two-layer nonlinear fully
   connected projectors", so the prompt carries the instruction only.
2. The triplet-derived keys (``observation.consequence_state`` and
   ``observation.context_actions``) are registered as derived features so the standard
   normalizer scales them with the same statistics as ``observation.state`` and ``action``.
   Context actions must live in the same normalized space as the model's own outputs,
   because at deployment the buffer is filled with what the policy itself predicted.
"""

from __future__ import annotations

from typing import Any

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_reflective_vla import ReflectiveVLAConfig
from .modeling_reflective_vla import OBS_CONSEQUENCE_STATE, OBS_CONTEXT_ACTIONS


def register_derived_features(config: ReflectiveVLAConfig) -> None:
    """Declare the triplet-derived keys so the shared normalizer picks them up.

    Only non-visual keys are added: consequence *images* must not become
    ``image_features`` or they would be mistaken for extra cameras.
    """
    if config.num_context_triplets == 0:
        return
    state_feature = config.input_features.get(OBS_STATE)
    if state_feature is not None:
        config.input_features.setdefault(
            OBS_CONSEQUENCE_STATE,
            PolicyFeature(type=FeatureType.STATE, shape=state_feature.shape),
        )
    action_feature = config.output_features.get(ACTION)
    if action_feature is not None:
        config.input_features.setdefault(
            OBS_CONTEXT_ACTIONS,
            PolicyFeature(type=FeatureType.ACTION, shape=action_feature.shape),
        )


def alias_derived_stats(
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
) -> dict[str, dict[str, torch.Tensor]] | None:
    """Point the derived keys at the statistics of the key they are copies of."""
    if dataset_stats is None:
        return None
    stats = dict(dataset_stats)
    if OBS_STATE in stats:
        stats.setdefault(OBS_CONSEQUENCE_STATE, stats[OBS_STATE])
    if ACTION in stats:
        stats.setdefault(OBS_CONTEXT_ACTIONS, stats[ACTION])
    return stats


def make_reflective_vla_pre_post_processors(
    config: ReflectiveVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    register_derived_features(config)
    stats = alias_derived_stats(dataset_stats)

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )
    steps = make_default_policy_processor_steps(config, stats)

    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        relative_step,
        steps.normalize,
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            # The paper projects proprioception into the token space instead.
            include_state_in_prompt=not config.state_as_token,
        ),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        steps.to_device,
    ]

    from lerobot.processor import AbsoluteActionsProcessorStep

    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
