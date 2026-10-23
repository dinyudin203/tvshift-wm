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
"""Configuration for the goal-image-conditioned Inverse Dynamics Model (IDM).

This is a thin subclass of :class:`DiffusionConfig` that turns the standard Diffusion
Policy into a goal-conditioned IDM ``π(a_t | o_t, o_{t+H})``, following the low-level
recipe of SuSIE (Black et al., 2024): a short executed action chunk plus a goal image
sampled at a *variable* future offset (the ``delta_goals`` relabeling strategy).

The goal image is obtained at training time by appending a window of future frame
offsets to ``observation_delta_indices`` and selecting one of them per sample inside the
policy (see ``modeling_diffusion_idm.py``). At inference time the goal image is supplied
externally (an oracle subgoal from a reference demo in Phase 1, or a generated subgoal in
Phase 2).
"""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("diffusion_idm")
@dataclass
class DiffusionIDMConfig(DiffusionConfig):
    """Goal-image-conditioned IDM built on top of Diffusion Policy.

    Args:
        goal_offset_min: Minimum future offset (in frames) of the goal image relative to
            the current observation. At training a goal offset is sampled uniformly in
            ``[goal_offset_min, goal_offset_max]`` (SuSIE ``delta_goals`` semantics).
        goal_offset_max: Maximum future offset (in frames) of the goal image. Set
            ``goal_offset_min == goal_offset_max`` for a fixed goal horizon.
        use_separate_goal_encoder: If True, encode the goal image(s) with a dedicated RGB
            encoder instead of sharing the current-observation encoder. Recommended, as the
            goal image plays a different role than the current observation.

    Notes:
        - ``horizon`` must be a multiple of ``2 ** len(down_dims)`` (default 8). The
          SuSIE-style "action prediction horizon of 4" maps to ``n_action_steps=4`` with a
          modest ``horizon`` such as 16.
        - The goal image uses the *same* camera keys as the current observation (so it is
          normalized with the same image statistics); no extra dataset features are
          required for training.
    """

    # --- Goal conditioning (SuSIE delta_goals + goal-aligned action masking) ---
    # Goal candidates are range(min, max+1, step) = [1..8]. The policy samples one per item as the
    # goal offset H; training then supervises ONLY the first H actions a[t..t+H-1] (those that drive
    # o_t -> o_{t+H}) and masks a[t+H..t+horizon-1] out of the loss (see modeling_diffusion_idm.forward
    # goal-aligned action mask). Inference uses H = goal_offset_max (=8).
    goal_offset_min: int = 1
    goal_offset_max: int = 8
    goal_offset_step: int = 1
    # Sampling weight concentrated on the LARGEST offset (H = goal_offset_max). At inference the
    # policy almost always sees a goal exactly `n_action_steps` (=8) ahead (open-loop demo clock),
    # so a flat Uniform{1..8} over-samples short H that the policy rarely faces at test time. With
    # peak_prob=p the max offset gets probability p and the remaining (n_goal_candidates-1) offsets
    # SHARE (1-p) uniformly — concentrating training near the inference distribution while still
    # exposing the short-H "finishing" cases (gripper open / place) near the episode end.
    # peak_prob=0.0 recovers flat Uniform{1..8}; peak_prob=1.0 fixes H=goal_offset_max.
    goal_offset_peak_prob: float = 0.6
    use_separate_goal_encoder: bool = True
    # Restrict which camera image keys the policy uses (for BOTH current obs and goal).
    # None = use all VISUAL input features. For libero_goal we set this to the agentview
    # camera only: the wrist-camera goal image is highly end-effector-pose dependent, so it
    # is a noisy/unstable goal-conditioning signal. Using agentview only also halves the
    # number of RGB encoders (faster).
    image_keys: tuple[str, ...] | None = None

    # --- IDM-friendly overrides of the Diffusion Policy defaults ---
    # n_obs_steps=1: single current frame o_t (matches SuSIE gc_ddpm_bc obs_horizon=1; action_delta
    # indices become [0..horizon-1] = a[t..t+7], no wasted trivial a[t-1]).
    # horizon=8 + n_action_steps=8: predict an 8-step chunk and execute it whole (receding-8, no
    # discarded predictions). At inference give (o_t, o_{t+8}) and run all 8 steps before re-planning.
    n_obs_steps: int = 1
    horizon: int = 8
    n_action_steps: int = 8
    # REQUIRED for goal-aligned masking: the base DiffusionConfig defaults this to False, which
    # makes compute_loss ignore `action_is_pad` entirely. We set True so the forward()'s
    # goal-aligned mask (a[t+H..] excluded) AND episode-boundary padding are actually dropped
    # from the loss.
    do_mask_loss_for_padding: bool = True
    # 0 => sample the current frame o_t over the FULL episode, including the last frames. This
    # puts the *finishing* motion (placing the object / opening the gripper — where LIBERO success
    # is decided) IN-distribution, and also exposes the policy to the near-end goal=last-frame
    # (small effective H) case it faces at the end of a closed-loop rollout. The trailing horizon
    # actions that run past the episode end are copy-padded and masked out of the loss
    # (do_mask_loss_for_padding), and only the first n_action_steps are ever executed.
    drop_n_last_frames: int = 0

    # --- CoTracker point-tracking conditioning (optional extra input) ---
    # When True, a FROZEN CoTracker3 backbone encodes the pair (o_t, o_{t+H}) into per-point
    # correspondence tokens; these are pooled + projected to `cotracker_feature_dim` and APPENDED
    # to the diffusion global conditioning, alongside the ResNet image/goal embeddings. Only the
    # small projection head is trained. False reproduces the plain diffusion_idm baseline exactly.
    use_cotracker_cond: bool = False
    cotracker_ckpt: str = "/root/.cache/torch/hub/checkpoints/scaled_offline.pth"
    cotracker_repo_path: str = "/workspace/yujin/co-tracker"
    cotracker_window_len: int = 60
    cotracker_grid_size: int = 32          # G -> G*G query points tracked o_t -> o_{t+H}
    cotracker_iters: int = 4
    cotracker_img_size: int = 256          # backbone runs at this resolution (mult. of 4)
    cotracker_feature_dim: int = 256       # projected dim appended to the global conditioning
    # Perceiver pooling of the (N points x 2 frames) point-track tokens -> M learned latents
    # (cross-attention), then projected to `cotracker_feature_dim`. Replaces naive mean pooling so
    # the conditioning can attend to the most informative tracked points rather than averaging them.
    cotracker_pool_latents: int = 8
    cotracker_pool_heads: int = 6
    cotracker_pool_layers: int = 2
    # MEAN_STD denorm so the (already image-normalized) policy input is mapped back to the [0,255]
    # range the CoTracker backbone expects. Defaults = libero_goal_image agentview image stats;
    # OVERRIDE these to match your dataset's `observation.images.*` mean/std if different.
    cotracker_denorm_mean: tuple[float, float, float] = (0.4906, 0.4595, 0.4190)
    cotracker_denorm_std: tuple[float, float, float] = (0.2273, 0.2100, 0.1933)

    # --- D4RT 4D reconstruction/tracking conditioning (optional extra input) ---
    # When True, a FROZEN OpenD4RT video model encodes the 2-frame clip (o_t, o_{t+H}) into its
    # per-token scene/memory features (the output of `D4RTModel.encode_video`); these tokens are
    # Perceiver-pooled + projected to `d4rt_feature_dim` and APPENDED to the diffusion global
    # conditioning, exactly like the CoTracker feature. Only the pool + projection head trains.
    # False reproduces the plain diffusion_idm baseline. Mutually orthogonal to use_cotracker_cond
    # (both can be on, each adds its own feature block).
    use_d4rt_cond: bool = False
    # OpenD4RT repo (added to sys.path so `import src.model` resolves) + its model.yaml/checkpoint.
    # The .ckpt must be downloaded from HF (huggingface.co/Lijiaxin0111/OpenD4RT) into the
    # checkpoint dir; only model.yaml ships with the repo.
    d4rt_repo_path: str = "/workspace/yujin/Open-d4rt"
    d4rt_model_config: str = (
        "/workspace/yujin/Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml"
    )
    d4rt_ckpt: str = (
        "/workspace/yujin/Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt"
    )
    d4rt_img_size: int = 256                # backbone runs at this resolution (mult. of patch 16)
    d4rt_feature_dim: int = 256             # projected dim appended to the global conditioning
    # Frozen-backbone compute precision. True => run encode_video under bf16 autocast (~3-4x faster;
    # measured feature error vs fp32 is ~0.6% / cosine 0.99998, negligible since the trainable
    # pool/proj adapt to it). bf16 is safe for D4RT: same exponent range as fp32 (no overflow) and
    # the encoder has no large additive attention masks. False => fp32 (autocast disabled).
    d4rt_backbone_bf16: bool = True
    # Perceiver pool of the D4RT memory tokens -> M learned latents, then projected to feature_dim
    # (same head as the CoTracker conditioning).
    d4rt_pool_latents: int = 8
    d4rt_pool_heads: int = 8                # must divide the D4RT token width (1280 for vit-g/48CLIP)
    d4rt_pool_layers: int = 2
    # MEAN_STD denorm so the (image-normalized) policy input is mapped back to [0,1] (the range the
    # D4RT video encoder expects, i.e. raw RGB / 255). Defaults = libero_goal_image agentview stats.
    d4rt_denorm_mean: tuple[float, float, float] = (0.4906, 0.4595, 0.4190)
    d4rt_denorm_std: tuple[float, float, float] = (0.2273, 0.2100, 0.1933)

    # --- Conditioning ablation toggles (language / proprio) -----------------------------------
    # Goal-image-only base: set use_proprio_cond=false AND use_lang_cond=false. Each can be
    # turned on independently to study its contribution (the 3 variants: +lang, +proprio, +both).
    #
    # Proprioception (observation.state). True = the robot state is part of the global
    # conditioning (the stock Diffusion Policy behavior). False = the state input is zeroed so
    # the policy receives NO proprio signal (dims are kept so checkpoints stay shape-compatible).
    use_proprio_cond: bool = True
    # Language: a FROZEN CLIP instruction embedding (looked up from `lang_emb_path` by the
    # batch["task"] string) is projected to `lang_feature_dim` and APPENDED to the global
    # conditioning. Requires batch["task"] at both train and inference time.
    use_lang_cond: bool = False
    lang_emb_path: str = "/workspace/data/libero_goal_lang_emb.pt"
    lang_dim: int = 512                    # CLIP ViT-B/32 text projection dim (input to lang_proj)
    lang_feature_dim: int = 128            # projected dim appended to the global conditioning

    def __post_init__(self):
        super().__post_init__()
        if self.goal_offset_min < 1:
            raise ValueError(f"`goal_offset_min` must be >= 1. Got {self.goal_offset_min}.")
        if self.goal_offset_max < self.goal_offset_min:
            raise ValueError(
                "`goal_offset_max` must be >= `goal_offset_min`. "
                f"Got min={self.goal_offset_min}, max={self.goal_offset_max}."
            )
        if not 0.0 <= self.goal_offset_peak_prob <= 1.0:
            raise ValueError(
                f"`goal_offset_peak_prob` must be in [0, 1]. Got {self.goal_offset_peak_prob}."
            )

    @property
    def image_features(self) -> dict:
        """VISUAL input features, optionally restricted to ``image_keys`` (obs + goal)."""
        feats = super().image_features
        if self.image_keys is None:
            return feats
        missing = [k for k in self.image_keys if k not in feats]
        if missing:
            raise ValueError(f"`image_keys` {missing} not found among visual inputs {list(feats)}.")
        return {k: v for k, v in feats.items() if k in self.image_keys}

    @property
    def n_goal_candidates(self) -> int:
        """Number of candidate future frames loaded per sample for goal selection."""
        return len(range(self.goal_offset_min, self.goal_offset_max + 1, self.goal_offset_step))

    @property
    def observation_delta_indices(self) -> list:
        """Observation frame offsets to load per sample.

        Layout: the first ``n_obs_steps`` entries are the current-observation history
        ``[1 - n_obs_steps, ..., 0]``; the remaining entries are the goal-candidate window
        ``range(goal_offset_min, goal_offset_max + 1, goal_offset_step)``. The policy splits
        the loaded time dimension at ``n_obs_steps``.
        """
        history = list(range(1 - self.n_obs_steps, 1))
        goal_window = list(range(self.goal_offset_min, self.goal_offset_max + 1, self.goal_offset_step))
        return history + goal_window
