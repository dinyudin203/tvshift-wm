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
"""Configuration for In-Context World Modeling (ICWM).

Implements the policy config for "In-Context World Modeling for Robotic Control"
(arXiv 2606.26025). ICWM reformulates the standard VLA policy pi_theta(a_t | o_t, l) as

    a_t ~ pi_theta(a_t | Psi(T), o_t, l)

where T = {(o^s_i, a_i, o^e_i)}_{i=1..N} is a set of N task-agnostic "interaction clips"
collected by randomly probing the environment/robot under a fixed system configuration
(camera viewpoint, morphology, ...). Psi is *not* a separate network: it is realized by
simply prepending the N interaction clips (as image + FAST action tokens) to the front of
the input sequence and letting the shared backbone transformer attend over everything
causally in a single forward pass (see paper Sec 4, Fig 2).

Design notes / deviations from a literal reading of the paper, forced by codebase reuse:
  * Backbone: plain dense `transformers.Qwen2_5_VLForConditionalGeneration` (NOT the
    Mixture-of-Experts `Qwen2_5_VLMoEForAction` used by lerobot's `wall_x` policy) -- wired
    the same way `lerobot.policies.eo1` wires its (flow-matching) Qwen2.5-VL backbone.
  * Action tokenization: FAST (DCT + BPE) via a HF `AutoProcessor` action tokenizer
    (default `lerobot/fast-action-tokenizer`, the same one `pi0_fast`/`wall_x` use), with the
    resulting discrete token ids added to the Qwen tokenizer vocabulary as
    `<|action_token_i|>` special tokens -- i.e. reusing/extending the shared vocabulary,
    following the same convention as `lerobot.policies.wall_x` (see
    `wall_x/modeling_wall_x.py::define_action_token_id`) rather than PI0-FAST's trick of
    reusing the *top* of an existing vocabulary (which only works because PaliGemma's
    tokenizer already has enough unused ids at the top -- Qwen's does not reserve such a
    block, so it is safer/more literal to add genuinely new tokens and resize the embedding
    matrix, exactly as wall_x already does for its own MoE backbone).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )
else:
    Qwen2_5_VLConfig = None
    Qwen2_5_VLTextConfig = None
    Qwen2_5_VLVisionConfig = None


@PreTrainedConfig.register_subclass("icwm")
@dataclass
class ICWMConfig(PreTrainedConfig):
    """Configuration for the ICWM policy (In-Context World Modeling)."""

    # --- Backbone -----------------------------------------------------------------
    # Plain dense Qwen2.5-VL-3B, NOT a MoE variant (paper fidelity requirement).
    backbone_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_config: dict | None = None
    image_min_pixels: int | None = 64 * 28 * 28
    image_max_pixels: int | None = 128 * 28 * 28
    use_fast_processor: bool = False
    dtype: str = "auto"  # "auto" | "bfloat16" | "float32"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = False

    # --- In-context world modeling ---------------------------------------------------
    # N interaction clips T = {(o^s_i, a_i, o^e_i)}, i = 1..N, prepended to the task query.
    n_context_clips: int = 5
    # Root directories (LeRobot v3.0 dataset roots) to pool interaction-clip segments from,
    # e.g. all 8 in-domain training azimuths for one LIBERO suite. When set (and
    # n_context_clips > 0), `lerobot.datasets.factory.make_dataset` wraps the training
    # dataset with `lerobot.datasets.icwm_context_sampler.ICWMContextSampledDataset` built
    # from a pool over these roots. If left empty, training falls back to n_context_clips=0
    # behavior (no ICWM_CONTEXT_* keys attached) regardless of the value above.
    context_pool_dataset_roots: list[str] = field(default_factory=list)
    context_pool_stride: int = 20
    # True: queries are drawn from the union of `context_pool_dataset_roots` and each query's clips
    # come from its OWN viewpoint's dataset (T carries information about the query's psi).
    # False: legacy behaviour (query = --dataset.repo_id only, clips from the pool of all viewpoints).
    context_match_viewpoint: bool = False
    # Where the N clips come from (with context_match_viewpoint=True):
    #   "pool"    : random segments of other episodes of the query's viewpoint (ICWM paper).
    #   "history" : the query's OWN episode, the N chunks just before it, oldest first (Reflective-VLA
    #               style context, arXiv triplets (O_i, A_i, O_{i+H})). Clip k starts H + g frames
    #               before clip k+1 (the last clip H + g before the query), g ~ U[0, context_history_max_gap]
    #               so the stride cannot be used as a shortcut; clips that would start before the
    #               episode are blank (black frames, arm still, gripper open), as at inference.
    #   "sametask_past": N clips from OTHER episodes of the query's own task (same viewpoint), each at
    #               a random segment that ends at or before the query's step t, ordered by time.
    #   "probe"   : N self-exploration clips of the query's own scene (task + viewpoint), recorded with
    #               the evaluation's probing routine (gen_icwm_probe_pool.py) -> same context
    #               distribution in training and at test time. Needs context_probe_dir.
    #   "mixed"   : both, as two marked segments: N "pool" clips ("Exploration:", world model of the
    #               viewpoint) then context_mixed_n_history "history" clips ("History:", the episode so far).
    context_source: str = "pool"
    context_history_max_gap: int = 15
    context_probe_dir: str | None = None
    # "mixed" only: number of history clips, and the per-sample context dropout. With probability
    # context_drop_history_p only the exploration segment is kept, with probability
    # context_drop_explore_p only the history segment (mutually exclusive; otherwise both are kept).
    context_mixed_n_history: int = 5
    context_drop_history_p: float = 0.0
    context_drop_explore_p: float = 0.0

    # --- Action chunking / FAST tokenization ----------------------------------------
    action_chunk_size: int = 5  # paper: action chunk size = 5
    n_action_steps: int = 5
    max_action_dim: int = 32  # actions (context clips + target chunk) padded to this dim
    fast_tokenizer_name_or_path: str = "lerobot/fast-action-tokenizer"
    # Size of the discrete FAST/BPE action-token vocabulary. Each of these ids gets its own
    # new `<|action_token_i|>` entry appended to the Qwen tokenizer vocabulary (see module
    # docstring). 2048 matches `AutoProcessor.from_pretrained("lerobot/fast-action-tokenizer").vocab_size`
    # (verified directly against the real tokenizer, see icwm/README.md verification notes).
    action_token_vocab_size: int = 2048
    # Upper bound on the number of FAST tokens used to encode a single action chunk
    # (context clip action, or the target action chunk); sequences are padded/truncated to
    # this length so that batches have a static shape.
    max_action_tokens_per_chunk: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # --- Optimizer / scheduler (paper Sec 5 / Table 5) -----------------------------
    # AdamW, weight_decay=1e-4, lr peak 5e-5, 50k step linear warmup then cosine decay.
    optimizer_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 50_000
    scheduler_decay_steps: int = 300_000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.action_chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than "
                f"action_chunk_size ({self.action_chunk_size})"
            )
        if self.n_context_clips < 0:
            raise ValueError(f"n_context_clips must be >= 0, got {self.n_context_clips}")
        if self.context_source not in ("pool", "history", "sametask_past", "probe", "mixed"):
            raise ValueError(f"context_source must be 'pool', 'history', 'sametask_past', 'probe' or 'mixed', "
                             f"got {self.context_source!r}")
        if self.context_drop_history_p + self.context_drop_explore_p > 1.0:
            raise ValueError("context_drop_history_p + context_drop_explore_p must be <= 1")
        if self.context_source == "probe" and not self.context_probe_dir:
            raise ValueError("context_source='probe' requires context_probe_dir")
        if self.context_source != "pool" and not self.context_match_viewpoint:
            raise ValueError(f"context_source={self.context_source!r} requires context_match_viewpoint=True")

        if self.vlm_config is None:
            require_package("transformers", extra="icwm")
            self.vlm_config = Qwen2_5_VLConfig.from_pretrained(self.backbone_name_or_path).to_dict()

    @property
    def vlm_backbone_config(self) -> "Qwen2_5_VLConfig":
        require_package("transformers", extra="icwm")
        config_dict = deepcopy(self.vlm_config)
        if self.attn_implementation is not None:
            config_dict["attn_implementation"] = self.attn_implementation
        return Qwen2_5_VLConfig(**config_dict)

    @property
    def text_config(self) -> "Qwen2_5_VLTextConfig":
        return self.vlm_backbone_config.text_config

    @property
    def vision_config(self) -> "Qwen2_5_VLVisionConfig":
        return self.vlm_backbone_config.vision_config

    def validate_features(self) -> None:
        """Validate and set up ICWM input/output features."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "ICWM policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
