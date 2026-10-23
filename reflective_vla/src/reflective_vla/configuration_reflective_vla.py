# Copyright 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Configuration for Reflective VLA (arXiv:2606.25215).

Defaults reproduce the paper: ``K=8`` context frames, chunk horizon ``C=10``, PaliGemma-3B
prefix, an action expert of hidden width 1024, per-triplet instruction re-insertion, learned
frame-boundary tokens, and full fine-tuning with a 0.1x learning-rate multiplier on the VLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

from .masking import CONTEXT_MODES

#: Table 4 of the paper (8x H20, batch size is per GPU).
PAPER_TRAINING_PRESETS = {
    "libero_reactive": {"steps": 90_000, "batch_size_per_gpu": 32, "context_frames": 1},
    "libero_reflective": {"steps": 50_000, "batch_size_per_gpu": 4, "context_frames": 8},
    "libero_plus_reactive": {"steps": 50_000, "batch_size_per_gpu": 32, "context_frames": 1},
    "libero_plus_reflective": {"steps": 90_000, "batch_size_per_gpu": 4, "context_frames": 8},
    "bridge_reactive": {"steps": 80_000, "batch_size_per_gpu": 32, "context_frames": 1},
    "bridge_reflective": {"steps": 160_000, "batch_size_per_gpu": 6, "context_frames": 4},
}


@PreTrainedConfig.register_subclass("reflective_vla")
@dataclass
class ReflectiveVLAConfig(PI05Config):
    """Reflective VLA on top of the pi0.5 Mixture-of-Transformers backbone.

    Setting ``context_frames=1`` yields the paper's *matched reactive baseline*: identical
    backbone, projectors, action expert, optimizer and data, with no historical triplets.
    """

    # --- in-context triplet configuration (Sec. 3.3, Appendix A.2) ------------------
    #: ``K``. The paper uses 8 on LIBERO/LIBERO-Plus and 4 on SimplerEnv-Bridge.
    context_frames: int = 8
    #: Table 3a ablation axis: ``"O"`` / ``"OA"`` / ``"OAO"``. ``"OAO"`` is the full model.
    context_mode: str = "OAO"
    #: Tokens produced by ``g_A`` per historical action chunk ("eight learned tokens").
    action_history_tokens: int = 8
    #: Re-insert the instruction inside every triplet block, as the paper's template does.
    language_per_frame: bool = True
    #: Emit the ``tau_i`` temporal-index token.
    use_temporal_index: bool = True
    #: Learned ``<|frame_start|>`` / ``<|action_start|>`` / ... markers. The Limitations
    #: section credits these with mitigating the past-chunk extrapolation shortcut.
    use_boundary_tokens: bool = True
    #: Put the instruction before the images inside a block. The paper's template is
    #: instruction-first (a Qwen chat template); PaliGemma is natively image-first and that
    #: is what the pi0.5 prefix expects, so image-first is the default here.
    language_first: bool = False
    #: Proprioception as a projected continuous token (paper) instead of pi0.5's
    #: discretized state string inside the prompt.
    state_as_token: bool = True

    #: ``O'_i = O_{i+C}``. ``None`` means "use ``chunk_size``", which is the paper's setting.
    consequence_offset: int | None = None

    # --- frame sampling (Appendix C.1) ---------------------------------------------
    #: Backward spacing between adjacent context frames is ``chunk_size`` plus a random gap
    #: drawn from ``[0, context_gap_max]``. This is a shortcut-prevention device, not a
    #: tuning knob: it stops the policy extrapolating the target from a fixed stride.
    #: 15 for the LIBERO family, 5 for SimplerEnv-Bridge.
    context_gap_max: int = 15
    #: Evaluation uses a deterministic fixed stride (gap 0).
    eval_context_gap: int = 0

    # --- pi0.5 overrides matching the paper ---------------------------------------
    chunk_size: int = 10  # C = 10 for LIBERO and SimplerEnv-Bridge (30 for real-world)
    n_action_steps: int = 10
    #: Width 1024 == "an action expert with hidden dimension 1024".
    action_expert_variant: str = "gemma_300m"
    paligemma_variant: str = "gemma_2b"  # PaliGemma-3B
    dtype: str = "bfloat16"
    #: The instruction alone is short once the state leaves the prompt, and the prompt is
    #: repeated K times, so the pi0.5 default of 200 would be mostly padding.
    tokenizer_max_length: int = 48

    # --- optimization (Appendix C.1) ----------------------------------------------
    optimizer_lr: float = 1e-4  # "action expert and projectors use a peak learning rate of 1e-4"
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    #: "the VLM uses a 0.1x multiplier".
    vlm_lr_multiplier: float = 0.1
    #: "held frozen for the initial period in Table 4" -- the paper never states the number
    #: of steps (Table 4 has only Steps / Batch-per-GPU / K columns). ``None`` falls back to
    #: ``scheduler_warmup_steps``, which is our reading of "the initial period".
    vlm_freeze_steps: int | None = None

    # --- efficiency escape hatch, off by default ----------------------------------
    #: Average-pool every image's tokens by this factor before they enter the prefix.
    #: 1 = paper behaviour; larger values shrink the K-block prefix for smoke tests.
    image_token_pool: int = 1

    #: Reuse VLM keys/values for completed triplets across control steps (Sec. 3.3).
    use_context_kv_cache: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.context_frames < 1:
            raise ValueError(f"context_frames (K) must be >= 1, got {self.context_frames}")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {CONTEXT_MODES}, got {self.context_mode!r}")
        if self.context_gap_max < 0:
            raise ValueError(f"context_gap_max must be >= 0, got {self.context_gap_max}")
        if self.image_token_pool < 1:
            raise ValueError(f"image_token_pool must be >= 1, got {self.image_token_pool}")
        if self.action_history_tokens < 1:
            raise ValueError(f"action_history_tokens must be >= 1, got {self.action_history_tokens}")
        if self.state_as_token and self.use_proprioceptive_memory:
            raise ValueError(
                "state_as_token and use_proprioceptive_memory (MEM) both project proprioception "
                "into the backbone; enable only one."
            )

    @property
    def num_context_triplets(self) -> int:
        """``K - 1`` -- the completed triplets that precede the query observation."""
        return self.context_frames - 1

    @property
    def effective_consequence_offset(self) -> int:
        return self.chunk_size if self.consequence_offset is None else self.consequence_offset

    @property
    def is_reactive(self) -> bool:
        """``K=1`` reproduces the matched reactive pi0.5 baseline."""
        return self.context_frames == 1

    @property
    def action_delta_indices(self) -> list:
        # Every sampled frame is also a prediction target, so a training item needs the
        # action chunk of each of the K frames. The dataset wrapper slices per frame; the
        # widest span it can ask for is K-1 backward strides plus one chunk.
        span = (self.context_frames - 1) * (self.chunk_size + self.context_gap_max)
        return list(range(-span, self.chunk_size))

    def get_scheduler_preset(self):
        from lerobot.optim import CosineDecayWithWarmupSchedulerConfig

        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def resolved_vlm_freeze_steps(self) -> int:
        return self.scheduler_warmup_steps if self.vlm_freeze_steps is None else self.vlm_freeze_steps
