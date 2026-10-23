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
"""Goal-image-conditioned Inverse Dynamics Model (IDM) built on Diffusion Policy.

Implements ``π(a_t | o_t, o_{t+H})``: the standard Diffusion Policy global conditioning
(current state + current images, over ``n_obs_steps``) is augmented with an encoding of a
**goal image** ``o_{t+H}`` that is appended to the global conditioning vector exactly once
(it has no time dimension).

Goal image plumbing
-------------------
- Training (``forward``): ``observation_delta_indices`` loads, for every camera key, the
  current-observation history followed by a window of future goal candidates (see
  ``DiffusionIDMConfig.observation_delta_indices``). The policy splits that time dimension
  at ``n_obs_steps`` and selects one goal candidate per sample uniformly at random
  (SuSIE ``delta_goals`` semantics → variable goal horizon H).
- Inference (``select_action``): the environment only provides the current observation, so
  the goal image must be supplied by the eval harness under the ``OBS_GOAL_IMAGES`` key,
  shaped ``(B, num_cameras, C, H, W)`` and already normalized.
"""

import sys

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .configuration_diffusion_idm import DiffusionIDMConfig

# Batch key holding the (already normalized) goal image(s), shape (B, num_cameras, C, H, W).
OBS_GOAL_IMAGES = "observation.goal_images"


class PerceiverPool(nn.Module):
    """Compress a variable-length token set ``(B, T, D)`` to ``M`` learned latents ``(B, M, D)``.

    ``M`` learned latent queries cross-attend to the input tokens, followed by ``layers`` blocks of
    latent self-attention + MLP (Perceiver-IO style). Lets the conditioning attend to the most
    informative tracked points instead of averaging them.
    """

    def __init__(self, dim: int, num_latents: int, heads: int = 6, layers: int = 2, mlp_ratio: int = 4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "norm1": nn.LayerNorm(dim),
                    "attn": nn.MultiheadAttention(dim, heads, batch_first=True),
                    "norm2": nn.LayerNorm(dim),
                    "mlp": nn.Sequential(
                        nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
                    ),
                }
            )
            for _ in range(layers)
        )

    def forward(self, tokens: Tensor) -> Tensor:
        b = tokens.shape[0]
        q = self.q_norm(self.latents.unsqueeze(0).expand(b, -1, -1))  # (B, M, D)
        kv = self.kv_norm(tokens)
        x, _ = self.cross_attn(q, kv, kv)  # (B, M, D)
        for blk in self.blocks:
            h = blk["norm1"](x)
            x = x + blk["attn"](h, h, h)[0]
            x = x + blk["mlp"](blk["norm2"](x))
        return x  # (B, M, D)


class CoTrackerCondEncoder(nn.Module):
    """Frozen CoTracker3 backbone -> pooled point-tracking conditioning vector for (o_t, o_{t+H}).

    Tracks a ``G x G`` query grid placed on ``o_t`` to the goal frame ``o_{t+H}`` and taps the
    updateformer's per-point token features (the same correspondence features the CoTracker-IDM
    uses). The N points x 2 frames tokens are compressed by a Perceiver pool to ``M`` latents and
    projected to ``out_dim``, giving a compact motion/correspondence descriptor appended to the
    diffusion global conditioning.

    The backbone is frozen (a fixed correspondence prior); only the Perceiver pool + projection
    head train. Inputs are the policy's MEAN_STD-normalized images; they are denormalized back to
    [0,255] (the range CoTracker expects) using the dataset image stats from the config.
    """

    def __init__(self, config: DiffusionIDMConfig):
        super().__init__()
        if config.cotracker_repo_path not in sys.path:
            sys.path.insert(0, config.cotracker_repo_path)
        from cotracker.models.build_cotracker import build_cotracker
        from cotracker.models.core.model_utils import get_points_on_a_grid

        self._grid_fn = get_points_on_a_grid
        self.backbone = build_cotracker(
            checkpoint=config.cotracker_ckpt, offline=True, window_len=config.cotracker_window_len
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self.grid_size = config.cotracker_grid_size
        self.iters = config.cotracker_iters
        self.img_size = config.cotracker_img_size
        self.out_dim = config.cotracker_feature_dim
        token_dim = 384  # updateformer token width (fixed by the backbone)

        self.register_buffer(
            "denorm_mean", torch.tensor(config.cotracker_denorm_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "denorm_std", torch.tensor(config.cotracker_denorm_std).view(1, 3, 1, 1)
        )
        # Perceiver-pool the (N points x 2 frames) tokens to M latents, then project to out_dim.
        self.num_latents = config.cotracker_pool_latents
        self.pool = PerceiverPool(
            token_dim, self.num_latents, config.cotracker_pool_heads, config.cotracker_pool_layers
        )
        self.proj = nn.Sequential(
            nn.Linear(token_dim * self.num_latents, self.out_dim), nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # keep the frozen backbone in eval (no BN/dropout drift)
        return self

    def _queries(self, b: int, h: int, w: int, device) -> Tensor:
        gp = self._grid_fn(self.grid_size, (h, w), device=device)  # [1, N, 2] (x, y)
        n = gp.shape[1]
        q = torch.cat([torch.zeros(1, n, 1, device=device), gp], dim=-1)  # frame-0 queries
        return q.expand(b, -1, -1).contiguous()

    def _denorm_to_255(self, x: Tensor) -> Tensor:
        """(B,3,H,W) MEAN_STD-normalized -> [0,255]; resize to the backbone resolution."""
        x = (x * self.denorm_std + self.denorm_mean) * 255.0
        x = x.clamp(0, 255)
        s = self.img_size
        if x.shape[-1] != s or x.shape[-2] != s:
            x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        return x

    def forward(self, o_t: Tensor, o_goal: Tensor) -> Tensor:
        b = o_t.shape[0]
        video = torch.stack([self._denorm_to_255(o_t), self._denorm_to_255(o_goal)], dim=1).float()
        q = self._queries(b, self.img_size, self.img_size, video.device)
        # Run the frozen backbone in fp32 (its large additive masks overflow under autocast/fp16).
        with torch.autocast(device_type=video.device.type, enabled=False), torch.no_grad():
            out = self.backbone(video, q, iters=self.iters, return_token_features=True)
        tokens = out[-1]                          # (B, N, 2, 384): N points x 2 frames
        tokens = tokens.reshape(b, -1, tokens.shape[-1])  # (B, N*2, 384) token set
        latents = self.pool(tokens)               # (B, M, 384) Perceiver compression
        return self.proj(latents.reshape(b, -1))  # (B, out_dim)


class D4RTCondEncoder(nn.Module):
    """Frozen OpenD4RT video model -> pooled 4D-scene conditioning vector for (o_t, o_{t+H}).

    Feeds the 2-frame clip ``[o_t, o_{t+H}]`` through ``D4RTModel.encode_video`` and taps the
    per-token memory features (the unified depth/correspondence/camera scene representation the
    D4RT decoder cross-attends to). The ``N`` memory tokens are compressed by a Perceiver pool to
    ``M`` latents and projected to ``out_dim``, giving a compact 4D motion/geometry descriptor
    appended to the diffusion global conditioning — the structural analog of ``CoTrackerCondEncoder``.

    The backbone is frozen (a fixed 4D prior); only the Perceiver pool + projection head train.
    Inputs are the policy's MEAN_STD-normalized images; they are denormalized back to the [0,1] RGB
    range the D4RT encoder expects (raw pixels / 255) using the dataset image stats from the config.
    """

    def __init__(self, config: DiffusionIDMConfig):
        super().__init__()
        if config.d4rt_repo_path not in sys.path:
            sys.path.insert(0, config.d4rt_repo_path)
        from src.core import load_yaml_config
        from src.model import build_model

        cfg = load_yaml_config(config.d4rt_model_config)
        self.backbone = build_model(cfg["model"])
        # Load the released weights (HF checkpoint). strict=False: encode_video only needs the
        # encoder + memory_proj; decoder/head tensors are loaded too but simply unused here.
        from src.core import load_checkpoint

        payload = load_checkpoint(config.d4rt_ckpt, map_location="cpu")
        state_dict = payload
        if isinstance(payload, dict):
            for k in ("state_dict", "model", "module", "network", "net"):
                if isinstance(payload.get(k), dict):
                    state_dict = payload[k]
                    break
        self.backbone.load_state_dict(state_dict, strict=False)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self.img_size = config.d4rt_img_size
        self.out_dim = config.d4rt_feature_dim
        self.backbone_bf16 = config.d4rt_backbone_bf16
        # Memory-token width = decoder hidden_dim (encode_video projects enc_hidden -> dec_hidden).
        token_dim = int(cfg.get_path("model.decoder.hidden_dim", cfg.get_path("model.encoder.hidden_dim", 1280)))

        self.register_buffer("denorm_mean", torch.tensor(config.d4rt_denorm_mean).view(1, 3, 1, 1))
        self.register_buffer("denorm_std", torch.tensor(config.d4rt_denorm_std).view(1, 3, 1, 1))
        self.num_latents = config.d4rt_pool_latents
        self.pool = PerceiverPool(
            token_dim, self.num_latents, config.d4rt_pool_heads, config.d4rt_pool_layers
        )
        self.proj = nn.Sequential(
            nn.Linear(token_dim * self.num_latents, self.out_dim), nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # keep the frozen backbone in eval (no dropout drift)
        return self

    def _denorm_to_01(self, x: Tensor) -> Tensor:
        """(B,3,H,W) MEAN_STD-normalized -> [0,1] RGB; resize to the backbone resolution."""
        x = x * self.denorm_std + self.denorm_mean
        x = x.clamp(0, 1)
        s = self.img_size
        if x.shape[-1] != s or x.shape[-2] != s:
            x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        return x

    def forward(self, o_t: Tensor, o_goal: Tensor) -> Tensor:
        b = o_t.shape[0]
        # D4RT expects video [B, T, C, H, W] in [0,1]; the 2-frame clip is (o_t, o_{t+H}).
        video = torch.stack([self._denorm_to_01(o_t), self._denorm_to_01(o_goal)], dim=1).float()
        # Frozen-backbone precision (config.d4rt_backbone_bf16): bf16 autocast => ~3-4x faster with
        # ~0.6% feature error vs fp32 (negligible); else fp32 with autocast disabled. The trainable
        # pool/proj adapt to whichever the backbone emits, so this is a pure speed/precision dial.
        use_bf16 = self.backbone_bf16 and video.device.type == "cuda"
        with torch.no_grad():
            with torch.autocast(
                device_type=video.device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                tokens = self.backbone.encode_video(video=video, aspect_ratio=None)  # (B, N, token_dim)
        # Hand fp32 tokens to the trainable pool; the outer training autocast recasts as needed.
        latents = self.pool(tokens.float())       # (B, M, token_dim) Perceiver compression
        return self.proj(latents.reshape(b, -1))  # (B, out_dim)


def _build_model(config: DiffusionIDMConfig):
    """Lazily import the base diffusion building blocks and build the IDM network."""
    from lerobot.policies.diffusion.modeling_diffusion import (
        DiffusionConditionalUnet1d,
        DiffusionModel,
        DiffusionRgbEncoder,
        _make_noise_scheduler,
    )

    class DiffusionIDMNet(DiffusionModel):
        """Diffusion model whose global conditioning also encodes a goal image."""

        def __init__(self, config: DiffusionIDMConfig):
            # Bypass DiffusionModel.__init__ (it builds a U-Net with the wrong
            # conditioning dim); build everything here with the goal encoder added.
            nn.Module.__init__(self)
            self.config = config

            if not config.image_features:
                raise ValueError("diffusion_idm requires at least one image feature for the goal image.")

            # --- Current-observation conditioning (per obs step) ---
            obs_cond_dim = config.robot_state_feature.shape[0]
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                obs_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                obs_cond_dim += self.rgb_encoder.feature_dim * num_images
            if config.env_state_feature:
                obs_cond_dim += config.env_state_feature.shape[0]

            # --- Goal-image conditioning (appended once, no time dimension) ---
            if config.use_separate_goal_encoder:
                if config.use_separate_rgb_encoder_per_camera:
                    goal_encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                    self.goal_rgb_encoder = nn.ModuleList(goal_encoders)
                    goal_feature_dim = goal_encoders[0].feature_dim
                else:
                    self.goal_rgb_encoder = DiffusionRgbEncoder(config)
                    goal_feature_dim = self.goal_rgb_encoder.feature_dim
            else:
                # Share the current-observation encoder for the goal image.
                self.goal_rgb_encoder = self.rgb_encoder
                goal_feature_dim = (
                    self.rgb_encoder[0].feature_dim
                    if isinstance(self.rgb_encoder, nn.ModuleList)
                    else self.rgb_encoder.feature_dim
                )
            goal_cond_dim = goal_feature_dim * num_images

            global_cond_dim = obs_cond_dim * config.n_obs_steps + goal_cond_dim

            # --- Optional CoTracker point-tracking conditioning for (o_t, o_{t+H}) ---
            self.cotracker_encoder = None
            if config.use_cotracker_cond:
                self.cotracker_encoder = CoTrackerCondEncoder(config)
                global_cond_dim += config.cotracker_feature_dim

            # --- Optional D4RT 4D-tracking conditioning for (o_t, o_{t+H}) ---
            self.d4rt_encoder = None
            if config.use_d4rt_cond:
                self.d4rt_encoder = D4RTCondEncoder(config)
                global_cond_dim += config.d4rt_feature_dim

            # --- Optional language conditioning (frozen CLIP instruction embedding) ---
            # Looked up by batch["task"] string from the precomputed table; projected and APPENDED
            # to the global conditioning. Only lang_proj is trained (the embeddings are frozen).
            self.lang_proj = None
            if config.use_lang_cond:
                self._build_lang_table(config)
                self.lang_proj = nn.Sequential(
                    nn.Linear(config.lang_dim, config.lang_feature_dim),
                    nn.GELU(),
                    nn.Linear(config.lang_feature_dim, config.lang_feature_dim),
                )
                global_cond_dim += config.lang_feature_dim

            self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)

            if config.compile_model:
                self.unet = torch.compile(self.unet, mode=config.compile_mode)

            self.noise_scheduler = _make_noise_scheduler(
                config.noise_scheduler_type,
                num_train_timesteps=config.num_train_timesteps,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                beta_schedule=config.beta_schedule,
                clip_sample=config.clip_sample,
                clip_sample_range=config.clip_sample_range,
                prediction_type=config.prediction_type,
            )

            if config.num_inference_steps is None:
                self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
            else:
                self.num_inference_steps = config.num_inference_steps

        def _encode_goal(self, goal_images: Tensor) -> Tensor:
            """Encode goal image(s) into a flat conditioning vector.

            Args:
                goal_images: (B, num_cameras, C, H, W)
            Returns:
                (B, num_cameras * goal_feature_dim)
            """
            batch_size = goal_images.shape[0]
            if isinstance(self.goal_rgb_encoder, nn.ModuleList):
                images_per_camera = einops.rearrange(goal_images, "b n ... -> n b ...")
                feats = [
                    encoder(images)
                    for encoder, images in zip(self.goal_rgb_encoder, images_per_camera, strict=True)
                ]
                return torch.cat(feats, dim=-1)
            # Shared encoder: fold the camera dim into the batch dim, then concatenate back.
            feats = self.goal_rgb_encoder(einops.rearrange(goal_images, "b n ... -> (b n) ..."))
            return einops.rearrange(feats, "(b n) f -> b (n f)", b=batch_size)

        def _build_lang_table(self, config):
            """Register {normalized instruction -> row} + a frozen [K, lang_dim] embedding buffer."""
            import re

            payload = torch.load(config.lang_emb_path, weights_only=False)
            table = payload["table"]                       # normalized instruction -> [dim] tensor
            self._lang_norm = lambda t: re.sub(r"\s+", " ", str(t).strip().lower())
            keys = sorted(table.keys())
            self._lang_key_to_row = {k: i for i, k in enumerate(keys)}
            mat = torch.stack([table[k].float() for k in keys], dim=0)  # [K, lang_dim]
            self.register_buffer("lang_table", mat, persistent=False)

        def _encode_lang(self, batch: dict[str, Tensor]) -> Tensor:
            """batch["task"] (list[str], len B) -> (B, lang_feature_dim)."""
            tasks = batch.get("task")
            if tasks is None:
                raise KeyError(
                    "use_lang_cond=True but batch has no 'task' key (the instruction string). "
                    "Ensure the dataset/env provides task language at train and inference time."
                )
            if isinstance(tasks, str):
                tasks = [tasks]
            rows = torch.as_tensor(
                [self._lang_key_to_row[self._lang_norm(t)] for t in tasks],
                device=self.lang_table.device,
            )
            emb = self.lang_table[rows]                    # (B, lang_dim), frozen
            return self.lang_proj(emb)                     # (B, lang_feature_dim)

        def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
            # Proprio ablation: zero the robot state so it carries no information (dims kept).
            if not self.config.use_proprio_cond:
                batch = {**batch, OBS_STATE: torch.zeros_like(batch[OBS_STATE])}
            obs_global = super()._prepare_global_conditioning(batch)  # (B, obs_cond_dim * n_obs_steps)
            goal_global = self._encode_goal(batch[OBS_GOAL_IMAGES])  # (B, goal_cond_dim)
            parts = [obs_global, goal_global]
            if self.cotracker_encoder is not None:
                # current frame o_t = last obs-history step, first camera; goal frame = o_{t+H}.
                o_t = batch[OBS_IMAGES][:, -1, 0]  # (B, C, H, W)
                o_goal = batch[OBS_GOAL_IMAGES][:, 0]  # (B, C, H, W)
                parts.append(self.cotracker_encoder(o_t, o_goal))  # (B, cotracker_feature_dim)
            if self.d4rt_encoder is not None:
                # current frame o_t = last obs-history step, first camera; goal frame = o_{t+H}.
                o_t = batch[OBS_IMAGES][:, -1, 0]  # (B, C, H, W)
                o_goal = batch[OBS_GOAL_IMAGES][:, 0]  # (B, C, H, W)
                parts.append(self.d4rt_encoder(o_t, o_goal))  # (B, d4rt_feature_dim)
            if self.lang_proj is not None:
                parts.append(self._encode_lang(batch))  # (B, lang_feature_dim)
            return torch.cat(parts, dim=-1)

    return DiffusionIDMNet(config)


class DiffusionIDMPolicy(PreTrainedPolicy):
    """Goal-image-conditioned IDM policy (Diffusion Policy backbone)."""

    config_class = DiffusionIDMConfig
    name = "diffusion_idm"

    def __init__(self, config: DiffusionIDMConfig, **kwargs):
        require_package("diffusers", extra="diffusion")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = _build_model(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`."""
        from collections import deque

        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    def _split_obs_and_goal(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Split the loaded observation time dimension into (current obs, goal image, goal offset).

        For each camera key the loaded tensor has shape ``(B, n_obs_steps + n_goal, C, H, W)``
        where the first ``n_obs_steps`` frames are the current-observation history and the
        rest are goal candidates. Stacks cameras, splits, and samples one goal per item.
        Also truncates the proprioceptive state keys to ``n_obs_steps`` in place.

        Returns:
            obs_images: (B, n_obs_steps, num_cameras, C, H, W)
            goal_images: (B, num_cameras, C, H, W)
            goal_offset: (B,) the sampled H for each item (frames from o_t to the goal frame).
        """
        n_obs = self.config.n_obs_steps
        # (B, T_total, num_cameras, C, H, W)
        stacked = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        obs_images = stacked[:, :n_obs]
        goal_candidates = stacked[:, n_obs:]  # (B, n_goal, num_cameras, C, H, W)

        batch_size, n_goal = goal_candidates.shape[0], goal_candidates.shape[1]
        if n_goal < 1:
            raise ValueError("No goal candidate frames were loaded; check `observation_delta_indices`.")
        # Sample one goal offset per item (variable H, SuSIE delta_goals). With
        # goal_offset_peak_prob=p>0 the LARGEST offset (last candidate, H=goal_offset_max) gets
        # probability p and the remaining offsets share (1-p) uniformly, matching the inference
        # distribution (goal almost always n_action_steps ahead). p=0 => flat Uniform.
        peak_prob = self.config.goal_offset_peak_prob
        if peak_prob > 0.0 and n_goal > 1:
            weights = torch.full(
                (n_goal,), (1.0 - peak_prob) / (n_goal - 1), device=goal_candidates.device
            )
            weights[-1] = peak_prob  # last candidate == goal_offset_max == max H
            sel = torch.multinomial(weights, batch_size, replacement=True)
        else:
            sel = torch.randint(0, n_goal, (batch_size,), device=goal_candidates.device)
        goal_images = goal_candidates[torch.arange(batch_size, device=goal_candidates.device), sel]
        # Map the candidate index back to the actual frame offset H = min + sel * step.
        goal_offset = self.config.goal_offset_min + sel * self.config.goal_offset_step  # (B,)

        # Truncate proprio/env-state to the current-observation history.
        batch[OBS_STATE] = batch[OBS_STATE][:, :n_obs]
        if self.config.env_state_feature and OBS_ENV_STATE in batch:
            batch[OBS_ENV_STATE] = batch[OBS_ENV_STATE][:, :n_obs]
        return obs_images, goal_images, goal_offset

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Training forward: build the goal image, then compute the diffusion loss.

        Goal-aligned action masking: the goal is o_{t+H}, so only the first H actions
        ``a[t..t+H-1]`` (which drive o_t -> o_{t+H}) are supervised; actions ``a[t+H..t+horizon-1]``
        are masked out of the loss (OR'd into ``action_is_pad`` alongside any episode-boundary pad).
        """
        if not self.config.image_features:
            raise ValueError("diffusion_idm requires image features.")
        batch = dict(batch)  # shallow copy
        obs_images, goal_images, goal_offset = self._split_obs_and_goal(batch)
        batch[OBS_IMAGES] = obs_images
        batch[OBS_GOAL_IMAGES] = goal_images

        # Mask every action at index k >= H (beyond the goal) out of the loss.
        horizon = self.config.horizon
        k = torch.arange(horizon, device=goal_offset.device)  # (horizon,)
        beyond_goal = k[None, :] >= goal_offset[:, None]  # (B, horizon), True where k >= H
        if "action_is_pad" in batch:
            batch["action_is_pad"] = batch["action_is_pad"] | beyond_goal
        else:
            batch["action_is_pad"] = beyond_goal

        loss = self.diffusion.compute_loss(batch)
        return loss, None

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        goal_images = batch[OBS_GOAL_IMAGES]
        stacked = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        stacked[OBS_GOAL_IMAGES] = goal_images
        # "task" (instruction string) isn't an observation queue key, so the rebuild above drops
        # it; carry it through for use_lang_cond=True (see _encode_lang).
        if "task" in batch:
            stacked["task"] = batch["task"]
        return self.diffusion.generate_actions(stacked, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Inference: the goal image must be provided by the harness under ``OBS_GOAL_IMAGES``.

        Args:
            batch: current environment observation plus ``OBS_GOAL_IMAGES`` of shape
                ``(B, num_cameras, C, H, W)`` (already normalized).
        """
        if ACTION in batch:
            batch.pop(ACTION)
        if OBS_GOAL_IMAGES not in batch:
            raise KeyError(
                f"diffusion_idm requires a goal image under '{OBS_GOAL_IMAGES}' at inference time; "
                "the eval harness must supply it."
            )
        batch = dict(batch)
        goal_images = batch.pop(OBS_GOAL_IMAGES)
        if self.config.image_features:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk({**batch, OBS_GOAL_IMAGES: goal_images}, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()
