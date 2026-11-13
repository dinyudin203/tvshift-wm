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

    # Per-query camera ray conditioning; RGB + 6 Plucker channels at each vision patch.
    use_plucker: bool = False
    plucker_convention: str = "opengl"
    plucker_channel_order: str = "moment_direction"
    # "zero": plucker_proj starts at 0 (untrained policy == RGB-only Qwen). "random": nn.Linear default init,
    # as AxisGuide's plucker_concat leaves its added patch-conv channels, so rays perturb features from step 0.
    plucker_init: str = "zero"
    plucker_calibration_dir: str = ""
    # Where the rays enter (arXiv 2510.02268 Fig. 2). "early": plucker_proj(ray patch) added to the ViT patch
    # embedding, i.e. a 9-channel first layer (the paper's variant for encoders trained from scratch).
    # "late": the paper's variant for pretrained encoders -- the Qwen vision tower sees RGB only; a small conv
    # net encodes the ray map per merged token and is concatenated channel-wise with the tower output, then a
    # linear layer (identity on the image part, zero on the ray part at init) maps it back to the LM width.
    # plucker_init only applies to "early".
    plucker_fusion: str = "early"
    # Channels of the ray encoder output. 0 = the image latent's own width (out_hidden_size, 2048 for 3B), as in
    # the paper: "encode the Plucker map to the same dimension as the image latent", then concatenate.
    plucker_late_channels: int = 0
    # Late fusion only. "concat" (default, as trained so far): Linear([latent | ray_feat]) initialised to [I | 0];
    # its image block is free to drift away from identity. "residual": latent + Linear(ray_feat) -- the image
    # path stays exactly the pretrained tower output (equivalent to concat with the image block frozen at I).
    plucker_late_mode: str = "concat"
    # Std of the normal init of the ray block of the fusion layer. 0 = zero init (the ray CNN then gets no
    # gradient until that block has moved away from 0).
    plucker_late_init_std: float = 0.0
    # Peak learning rate of the late-fusion module (ray CNN + fusion layer); 0 = optimizer_lr like the backbone.
    # Same warmup/decay schedule (it scales every parameter group).
    plucker_late_lr: float = 0.0
    # Frame the Plucker rays are expressed in. "world": MuJoCo world (as trained so far). "base": robot base
    # (robot0_link0) frame, the frame of the delta-EEF actions; the same camera angle then gives the same rays
    # in every LIBERO suite. Needs observation.robot_base_to_world (calibration sidecar robot_base_to_world.npy
    # at training, the simulator at evaluation).
    plucker_frame: str = "world"
    # Keep the whole Qwen vision tower (patch embed, ViT blocks, merger) at its pretrained weights; the LM,
    # the Plucker late-fusion module and the rest still train.
    freeze_vision_encoder: bool = False

    # Camera reasoning step: no ray input; the policy first infers query-view Plucker maps from the context,
    # as slot tokens at the start of its answer, then decodes the action tokens that attend to those slots.
    # Calibration is the regression target during training and is not needed at inference.
    plucker_reasoning: bool = False
    plucker_reasoning_grid: int = 4  # slots per query view = grid x grid
    plucker_reasoning_loss_weight: float = 1.0

    # Camera tokens: the vision encoder stays the pretrained Qwen one. Each query view gets grid x grid
    # <|camera_slot_i|> tokens placed right before its image; a token's input embedding is its own (learned)
    # token embedding plus camera_token_mlp(Plucker ray at that grid cell's center). Calibration is an input
    # at training and inference, exactly as for use_plucker.
    use_camera_tokens: bool = False
    camera_token_grid: int = 4
    camera_token_hidden: int = 512

    # Pixel-axis motion reasoning (motion_reasoning.py): before acting, the policy writes the delta-EEF
    # (robot-base frame) that would move the gripper motion_reasoning_cm along image +x / +y / +z in the
    # query camera, as text numbers, e.g. "x:+0.92,+0.01,-0.10;y:...;z:...;". Camera calibration
    # (plucker_calibration_dir, plucker_frame=base) is the target at training, not an input.
    #   mode "molmoact": answer = [reasoning text][FAST action tokens], one next-token CE over both,
    #                    so the action loss trains the VLM (MolmoAct-style reason-then-act).
    #   mode "ki"      : knowledge insulation (LAP / pi0.5-KI): the VLM gets the reasoning CE (+ the FAST
    #                    action CE when ki_vlm_fast_loss); a flow-matching action expert reads the VLM's
    #                    hidden states of prefix + reasoning through stop-gradient and its actions are the
    #                    ones executed.
    motion_reasoning: bool = False
    motion_reasoning_mode: str = "molmoact"
    motion_reasoning_cm: float = 5.0
    motion_reasoning_units: str = "chunk_step"  # context_step | chunk_step | total | cm
    # m of EEF travel per raw action unit per step, (x, y, z); least squares on the pair5 data (95k 5-step windows)
    motion_reasoning_gain: tuple[float, float, float] = (0.01092, 0.01245, 0.01072)
    motion_reasoning_decimals: int = 2
    motion_reasoning_loss_weight: float = 1.0
    motion_reasoning_max_tokens: int = 96
    ki_vlm_fast_loss: bool = True
    action_expert_width: int = 1024
    action_expert_depth: int = 8
    action_expert_heads: int = 16
    action_expert_loss_weight: float = 1.0
    action_expert_lr: float = 0.0  # 0 = optimizer_lr
    flow_num_steps: int = 10

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
    #   "explore" : self-exploration probing clips recorded at the query's viewpoint, in a scene of the
    #               query's task (build_icwm_explore_context.py: the eval's probe() run at training init
    #               states). N distinct clips of one probing session, as at eval.
    #   "sametask_past" : N clips from OTHER demos of the query's task (same viewpoint), each a random other demo and a
    #               random segment ending at or before the query's step t, time-ordered (tvshift-wm); clips that do
    #               not fit before t are left out (n_valid), none at all for t < H.
    #   "pool_sametask" : as "pool" but only demo segments of OTHER episodes of the query's own task (any timestep).
    #   "pool_past"     : as "sametask_past" but from OTHER episodes of ANY task at the viewpoint (segments ending by step t).
    context_source: str = "pool"
    context_history_max_gap: int = 15
    # context_source="explore": dir with <viewpoint dataset name>__{frames,actions}.npy / __meta.json
    context_explore_dir: str = ""
    # True: the context-clip actions get the same normalization (and stats) as the target action before
    # FAST tokenization, so a motion maps to the same tokens in the context and in the target. False keeps
    # the original behaviour (context actions tokenized raw) for checkpoints trained that way.
    normalize_context_actions: bool = False
    # Visual input features left out of the query (e.g. ["observation.images.wrist_image"]: agentview only).
    # Applied when the features are parsed from the dataset (policies/factory.py), so the saved config has them removed.
    drop_image_keys: list[str] = field(default_factory=list)
    # Cameras of every context-clip observation (pool / history / pool_sametask / pool_past clips), in this order:
    # each o^s / o^e becomes len(context_image_keys) consecutive images, e.g. [front, wrist] ->
    # [o^s front, o^s wrist, a, o^e front, o^e wrist]. Default: agentview only (the original format).
    context_image_keys: list[str] = field(default_factory=lambda: ["observation.images.image"])
    # Second context segment (needs context_match_viewpoint=True): the query's own episode, the
    # n_history_clips chunks just before it (as context_source="history"), placed after the
    # n_context_clips clips of context_source; the two segments are tagged "Exploration:" / "History:".
    # history_dropout: probability a training query gets no history at all, so the first segment
    # cannot be ignored in favour of copying the recent actions.
    n_history_clips: int = 0
    history_dropout: float = 0.0

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
        if self.use_plucker:
            if self.plucker_init not in ("zero", "random"):
                raise ValueError("plucker_init must be zero or random")
            if self.plucker_fusion not in ("early", "late"):
                raise ValueError("plucker_fusion must be early or late")
            if self.plucker_late_channels < 0:
                raise ValueError("plucker_late_channels must be >= 0 (0 = image latent width)")
            if self.plucker_late_mode not in ("concat", "residual"):
                raise ValueError("plucker_late_mode must be concat or residual")
            if self.plucker_late_init_std < 0 or self.plucker_late_lr < 0:
                raise ValueError("plucker_late_init_std and plucker_late_lr must be >= 0")
            if self.plucker_fusion != "late" and (self.plucker_late_mode != "concat" or self.plucker_late_init_std
                                                  or self.plucker_late_lr):
                raise ValueError("plucker_late_mode / plucker_late_init_std / plucker_late_lr need plucker_fusion=late")
            if self.plucker_frame not in ("world", "base"):
                raise ValueError("plucker_frame must be world or base")
            if self.n_context_clips != 0 or self.n_history_clips != 0:
                raise ValueError("Plucker conditioning currently supports ICWM no-context only")
            if self.plucker_convention not in ("opengl", "opencv"):
                raise ValueError("plucker_convention must be opengl or opencv")
            if self.plucker_channel_order not in ("moment_direction", "direction_moment"):
                raise ValueError("invalid Plucker channel order")
        if self.plucker_reasoning:
            if self.use_plucker:
                raise ValueError("plucker_reasoning infers cameras from context; disable calibrated use_plucker input")
            if self.n_context_clips <= 0:
                raise ValueError("plucker_reasoning requires n_context_clips > 0")
            if self.plucker_reasoning_grid < 1 or self.plucker_reasoning_loss_weight < 0:
                raise ValueError("invalid plucker_reasoning_grid or plucker_reasoning_loss_weight")
            if self.plucker_convention not in ("opengl", "opencv"):
                raise ValueError("plucker_convention must be opengl or opencv")
            if self.plucker_channel_order not in ("moment_direction", "direction_moment"):
                raise ValueError("invalid Plucker channel order")
        if self.use_camera_tokens:
            if self.use_plucker or self.plucker_reasoning:
                raise ValueError("use_camera_tokens replaces use_plucker / plucker_reasoning; enable only one")
            if self.camera_token_grid < 1 or self.camera_token_hidden < 1:
                raise ValueError("invalid camera_token_grid or camera_token_hidden")
            if self.plucker_convention not in ("opengl", "opencv"):
                raise ValueError("plucker_convention must be opengl or opencv")
            if self.plucker_channel_order not in ("moment_direction", "direction_moment"):
                raise ValueError("invalid Plucker channel order")
        if self.motion_reasoning:
            if self.use_plucker or self.plucker_reasoning or self.use_camera_tokens:
                raise ValueError("motion_reasoning infers the camera itself; disable use_plucker / plucker_reasoning / "
                                 "use_camera_tokens")
            if self.motion_reasoning_mode not in ("molmoact", "ki"):
                raise ValueError("motion_reasoning_mode must be molmoact or ki")
            if self.motion_reasoning_units not in ("context_step", "chunk_step", "total", "cm"):
                raise ValueError("motion_reasoning_units must be context_step, chunk_step, total or cm")
            if self.motion_reasoning_cm <= 0 or len(self.motion_reasoning_gain) != 3 or min(self.motion_reasoning_gain) <= 0:
                raise ValueError("invalid motion_reasoning_cm or motion_reasoning_gain")
            if self.plucker_convention != "opengl":
                raise ValueError("the calibration sidecars use OpenGL camera axes")
        if self.context_source not in ("pool", "pool_sametask", "pool_past", "history", "explore", "sametask_past"):
            raise ValueError(f"context_source must be 'pool', 'pool_sametask', 'pool_past', 'history', 'explore' or 'sametask_past', got {self.context_source!r}")
        if self.context_source in ("pool_sametask", "pool_past", "history", "explore", "sametask_past") and not self.context_match_viewpoint:
            raise ValueError(f"context_source={self.context_source!r} requires context_match_viewpoint=True")
        if self.n_history_clips > 0 and (not self.context_match_viewpoint or self.context_source == "history"):
            raise ValueError("n_history_clips > 0 needs context_match_viewpoint=True and context_source != 'history'")
        if self.context_source == "explore" and not self.context_explore_dir:
            raise ValueError("context_source='explore' requires context_explore_dir")

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
