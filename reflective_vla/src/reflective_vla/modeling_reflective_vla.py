# Copyright 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reflective VLA (arXiv:2606.25215) on top of the LeRobot pi0.5 backbone.

The architecture is pi0.5's Mixture-of-Transformers -- a PaliGemma prefix and a
flow-matching action expert sharing self-attention at every layer -- with the conditioning
interface replaced: instead of a single observation the prefix carries ``K-1`` completed
observation-action-consequence triplets followed by the current observation, and the suffix
carries one action-expert query slot per sampled frame.  A block-causal mask (Eq. 6) lets a
single forward pass supervise all ``K`` frames without any of them seeing its own executed
action or its own consequence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.common.vla_utils import (
    create_sinusoidal_pos_embedding,
    prepare_attention_masks_4d,
    resize_with_pad_torch,
)
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, get_gemma_config
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from .buffer import TripletBuffer
from .configuration_reflective_vla import ReflectiveVLAConfig
from .masking import (
    build_attention_mask,
    build_position_ids,
    build_prefix_attention_mask,
    build_prefix_layout,
    query_frames_for_inference,
    suffix_query_frames,
)
from .tokenization import PrefixSources, TripletProjectors, assemble_prefix, pool_image_tokens

OBS_CONSEQUENCE_IMAGES = "observation.consequence_images"
OBS_CONSEQUENCE_STATE = "observation.consequence_state"
OBS_CONTEXT_ACTIONS = "observation.context_actions"
OBS_FRAME_MASK = "observation.frame_mask"
OBS_TEMPORAL_INDEX = "observation.temporal_index"


class ReflectiveVLAPytorch(PI05Pytorch):
    """Core model: triplet prefix + one flow-matching query slot per sampled frame."""

    def __init__(self, config: ReflectiveVLAConfig):
        super().__init__(config)
        self.config: ReflectiveVLAConfig = config
        vlm_width = get_gemma_config(config.paligemma_variant).width
        self.triplet_projectors = TripletProjectors(
            width=vlm_width,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            chunk_size=config.chunk_size,
            action_history_tokens=config.action_history_tokens,
            use_boundary_tokens=config.use_boundary_tokens,
            use_temporal_index=config.use_temporal_index,
            state_as_token=config.state_as_token,
            temporal_min_period=config.min_period,
            temporal_max_period=config.max_period,
        )
        self._layout_cache: dict[tuple, object] = {}

    # ------------------------------------------------------------------ layout

    def build_layout(self, *, num_cameras: int, tokens_per_image: int, num_language_tokens: int):
        key = (num_cameras, tokens_per_image, num_language_tokens, self.config.context_frames)
        if key not in self._layout_cache:
            self._layout_cache[key] = build_prefix_layout(
                num_frames=self.config.context_frames,
                num_cameras=num_cameras,
                tokens_per_image=tokens_per_image,
                num_language_tokens=num_language_tokens,
                action_history_tokens=self.config.action_history_tokens,
                context_mode=self.config.context_mode,
                language_per_frame=self.config.language_per_frame,
                use_temporal_index=self.config.use_temporal_index,
                include_state=self.config.state_as_token,
                language_first=self.config.language_first,
                use_boundary_tokens=self.config.use_boundary_tokens,
            )
        return self._layout_cache[key]

    # ------------------------------------------------------------------- prefix

    def _embed_image_stack(self, images: Tensor) -> Tensor:
        """``[B, F, 3, H, W] -> [B, F, tokens_per_image, width]`` via the native VLM pathway."""
        bsize, num_frames = images.shape[:2]
        flat = images.flatten(0, 1)

        def image_embed_func(img):
            return self.paligemma_with_expert.embed_image(img)

        emb = self._apply_checkpoint(image_embed_func, flat)
        emb = emb.view(bsize, num_frames, emb.shape[-2], emb.shape[-1])
        return pool_image_tokens(emb, self.config.image_token_pool)

    def embed_triplet_prefix(
        self,
        *,
        obs_images: list[Tensor],
        obs_image_masks: list[Tensor],
        obs_states: Tensor,
        tokens: Tensor,
        language_masks: Tensor,
        frame_masks: Tensor,
        temporal_index: Tensor | None = None,
        consequence_images: list[Tensor] | None = None,
        consequence_image_masks: list[Tensor] | None = None,
        consequence_states: Tensor | None = None,
        context_actions: Tensor | None = None,
    ):
        """Assemble the full ``K``-block prefix.

        Returns ``(prefix_embs, prefix_pad_masks, layout)``.
        """
        cfg = self.config
        obs_embs = [self._embed_image_stack(img) for img in obs_images]
        tokens_per_image = obs_embs[0].shape[2]

        cons_embs: list[Tensor] = []
        if cfg.context_mode == "OAO" and cfg.num_context_triplets > 0:
            if consequence_images is None:
                raise ValueError("context_mode='OAO' requires consequence images")
            cons_embs = [self._embed_image_stack(img) for img in consequence_images]

        def lang_embed_func(tok):
            return self.paligemma_with_expert.embed_language_tokens(tok)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)

        obs_state_emb = self.triplet_projectors.embed_state(obs_states) if cfg.state_as_token else None
        cons_state_emb = (
            self.triplet_projectors.embed_state(consequence_states)
            if cfg.state_as_token and cfg.context_mode == "OAO" and cfg.num_context_triplets > 0
            else None
        )
        action_emb = (
            self.triplet_projectors.embed_action_history(context_actions)
            if "A" in cfg.context_mode and cfg.num_context_triplets > 0
            else None
        )
        temporal_emb = (
            self.triplet_projectors.embed_temporal_index(temporal_index)
            if cfg.use_temporal_index
            else None
        )
        boundaries = self.triplet_projectors.embed_boundaries() if cfg.use_boundary_tokens else None

        layout = self.build_layout(
            num_cameras=len(obs_embs),
            tokens_per_image=tokens_per_image,
            num_language_tokens=tokens.shape[1],
        )

        sources = PrefixSources(
            obs_images=obs_embs,
            obs_image_masks=obs_image_masks,
            frame_masks=frame_masks,
            language=lang_emb,
            language_mask=language_masks,
            obs_state=obs_state_emb,
            consequence_images=cons_embs,
            consequence_image_masks=consequence_image_masks or [],
            consequence_state=cons_state_emb,
            action_history=action_emb,
            temporal_index=temporal_emb,
            boundaries=boundaries,
        )
        prefix_embs, prefix_pads = assemble_prefix(layout, sources)
        return prefix_embs, prefix_pads, layout

    # ------------------------------------------------------------------- suffix

    def embed_query_slots(self, noisy_actions: Tensor, timestep: Tensor):
        """One flow-matching query slot per sampled frame.

        Args:
            noisy_actions: ``[B, Q, C, action_dim]``
            timestep: ``[B, Q]`` -- an independent flow time per slot, which AdaRMS supports
                because ``PiGemmaRMSNorm`` accepts a per-token conditioning tensor.

        Returns ``(suffix_embs [B, Q*C, width], suffix_pad_masks, adarms_cond [B, Q*C, width])``.
        """
        bsize, num_queries, chunk, _ = noisy_actions.shape
        width = self.action_in_proj.out_features

        def action_proj_func(x):
            return self.action_in_proj(x)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
        action_emb = action_emb.reshape(bsize, num_queries * chunk, width)

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            width,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(dtype=noisy_actions.dtype)

        def time_mlp_func(x):
            x = self.time_mlp_in(x)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)  # [B, Q, width]
        adarms_cond = time_emb.repeat_interleave(chunk, dim=1)  # [B, Q*C, width]

        pad = torch.ones(bsize, num_queries * chunk, dtype=torch.bool, device=noisy_actions.device)
        return action_emb, pad, adarms_cond

    # ------------------------------------------------------------------ training

    def forward(  # type: ignore[override]
        self,
        *,
        obs_images: list[Tensor],
        obs_image_masks: list[Tensor],
        obs_states: Tensor,
        tokens: Tensor,
        language_masks: Tensor,
        frame_masks: Tensor,
        actions: Tensor,
        noise: Tensor,
        time: Tensor,
        temporal_index: Tensor | None = None,
        consequence_images: list[Tensor] | None = None,
        consequence_image_masks: list[Tensor] | None = None,
        consequence_states: Tensor | None = None,
        context_actions: Tensor | None = None,
    ) -> Tensor:
        """One block-causal forward pass supervising all ``K`` frames.

        Args:
            actions: ``[B, K, C, action_dim]`` -- the target chunk of every sampled frame.
            noise / time: ``[B, K, C, action_dim]`` / ``[B, K]``.

        Returns per-element flow-matching losses of shape ``[B, K, C, action_dim]``.
        """
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions

        prefix_embs, prefix_pads, layout = self.embed_triplet_prefix(
            obs_images=obs_images,
            obs_image_masks=obs_image_masks,
            obs_states=obs_states,
            tokens=tokens,
            language_masks=language_masks,
            frame_masks=frame_masks,
            temporal_index=temporal_index,
            consequence_images=consequence_images,
            consequence_image_masks=consequence_image_masks,
            consequence_states=consequence_states,
            context_actions=context_actions,
        )
        suffix_embs, suffix_pads, adarms_cond = self.embed_query_slots(x_t, time)

        num_frames = self.config.context_frames
        chunk = self.config.chunk_size
        query_frames = suffix_query_frames(num_frames, chunk, device=prefix_pads.device)
        # A frame the episode could not fill is neither context nor target.
        suffix_pads = suffix_pads & frame_masks.repeat_interleave(chunk, dim=1)

        att_2d = build_attention_mask(layout, query_frames, prefix_pads, suffix_pads)
        position_ids = build_position_ids(layout, query_frames, prefix_pads, suffix_pads)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            adarms_cond = adarms_cond.to(dtype=torch.bfloat16)

        att_4d = prepare_attention_masks_4d(att_2d)

        def forward_func(prefix_embs, suffix_embs, att_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out.to(dtype=torch.float32).view(
            suffix_out.shape[0], num_frames, chunk, -1
        )

        def action_out_proj_func(x):
            return self.action_out_proj(x)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        # The projection may run in bf16 (FSDP mixed precision); compute the loss in fp32.
        return F.mse_loss(u_t.float(), v_t.float(), reduction="none")

    # ----------------------------------------------------------------- inference

    @torch.no_grad()
    def sample_actions(  # type: ignore[override]
        self,
        *,
        obs_images: list[Tensor],
        obs_image_masks: list[Tensor],
        obs_states: Tensor,
        tokens: Tensor,
        language_masks: Tensor,
        frame_masks: Tensor,
        temporal_index: Tensor | None = None,
        consequence_images: list[Tensor] | None = None,
        consequence_image_masks: list[Tensor] | None = None,
        consequence_states: Tensor | None = None,
        context_actions: Tensor | None = None,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Denoise the current frame's chunk given the triplet buffer.

        Only the last frame is a target at deployment, and the current block carries no
        evidence, so its query slot may read the whole valid prefix -- which is exactly
        pi0.5's suffix rule.  That lets the prefix be prefilled once into a KV cache and the
        denoising loop reuse pi0.5's :meth:`denoise_step` unchanged.
        """
        cfg = self.config
        num_steps = cfg.num_inference_steps if num_steps is None else num_steps
        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            noise = self.sample_noise((bsize, cfg.chunk_size, cfg.max_action_dim), device)

        prefix_embs, prefix_pads, layout = self.embed_triplet_prefix(
            obs_images=obs_images,
            obs_image_masks=obs_image_masks,
            obs_states=obs_states,
            tokens=tokens,
            language_masks=language_masks,
            frame_masks=frame_masks,
            temporal_index=temporal_index,
            consequence_images=consequence_images,
            consequence_image_masks=consequence_image_masks,
            consequence_states=consequence_states,
            context_actions=context_actions,
        )

        prefix_att_2d = build_prefix_attention_mask(layout, prefix_pads)
        prefix_position_ids = (torch.cumsum(prefix_pads.long(), dim=1) - 1).clamp(min=0)
        prefix_att_4d = prepare_attention_masks_4d(prefix_att_2d)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        from lerobot.policies.common.flow_matching import euler_integrate

        return euler_integrate(
            lambda input_x_t, current_timestep: self.denoise_step(
                prefix_pad_masks=prefix_pads,
                past_key_values=past_key_values,
                x_t=input_x_t,
                timestep=current_timestep,
            ),
            noise,
            num_steps,
        )

    def query_frames_at_inference(self, device) -> Tensor:
        return query_frames_for_inference(self.config.context_frames, self.config.chunk_size, device)


class ReflectiveVLAPolicy(PI05Policy):
    """LeRobot policy wrapper for Reflective VLA.

    Training batch contract (frames run oldest -> current along the frame axis)::

        observation.images.<cam>              [B, K,   3, H, W]   float in [0, 1]
        observation.consequence_images.<cam>  [B, K-1, 3, H, W]
        observation.state                     [B, K,   state_dim]
        observation.consequence_state         [B, K-1, state_dim]
        observation.context_actions           [B, K-1, C, action_dim]   executed chunks A_i
        observation.frame_mask                [B, K]      False = slot the episode cannot fill
        observation.temporal_index            [B, K]      backward offset in env steps
        action                                [B, K,   C, action_dim]   per-frame targets
        action_is_pad                         [B, K,   C]

    At inference the policy takes ordinary single-frame observations and builds the frame
    axis itself from :class:`~reflective_vla.buffer.TripletBuffer`.
    """

    config_class = ReflectiveVLAConfig
    name = "reflective_vla"

    def supports_rtc(self) -> bool:
        # The paper's suffix carries one query slot per frame; RTC's clean-prefix machinery
        # is defined for a single chunk and is not part of this reproduction.
        return False

    def __init__(self, config: ReflectiveVLAConfig, **kwargs):
        # PI05Policy.__init__ hard-codes ``PI05Pytorch``; go straight to the base class so we
        # build the triplet model once instead of instantiating a 3B backbone twice.
        from lerobot.policies.pretrained import PreTrainedPolicy
        from lerobot.utils.import_utils import require_package

        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.rtc_processor = None
        self.model = ReflectiveVLAPytorch(config)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    def reset(self):
        from collections import deque

        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        self._triplet_buffer = TripletBuffer(
            capacity=self.config.num_context_triplets, chunk_size=self.config.chunk_size
        )

    # ------------------------------------------------------------------ optimizer

    def get_optim_params(self):
        """Two parameter groups, as in Appendix C.1.

        "The action expert and projectors use a peak learning rate of 1e-4; the VLM uses a
        0.1x multiplier."
        """
        vlm_params, rest = [], []
        for name, param in self.named_parameters():
            (vlm_params if ".paligemma." in name else rest).append(param)
        return [
            {"params": rest, "lr": self.config.optimizer_lr},
            {"params": vlm_params, "lr": self.config.optimizer_lr * self.config.vlm_lr_multiplier},
        ]

    def set_training_step(self, step: int) -> None:
        """Apply the paper's VLM freeze schedule.

        "the VLM ... is held frozen for the initial period in Table 4, after which it
        follows the same linear warm-up and cosine decay schedule as the rest of the model."
        The paper never states the length of that period; :attr:`vlm_freeze_steps` exposes it
        and defaults to the warm-up length.
        """
        frozen = step < self.config.resolved_vlm_freeze_steps
        for name, param in self.named_parameters():
            if ".paligemma." in name:
                param.requires_grad = not frozen

    # ---------------------------------------------------------------- preprocess

    def _preprocess_image_stack(self, images: Tensor) -> Tensor:
        """``[B, F, ...] -> [B, F, 3, H, W]`` in ``[-1, 1]``, matching pi0.5 preprocessing."""
        device = next(self.parameters()).device
        img = images.to(device=device, dtype=torch.float32)
        if img.ndim == 4:
            img = img[:, None]
        channels_first = img.shape[2] == 3
        if channels_first:
            img = img.permute(0, 1, 3, 4, 2)
        if tuple(img.shape[2:4]) != tuple(self.config.image_resolution):
            bsize, num_frames = img.shape[:2]
            img = resize_with_pad_torch(
                img.flatten(0, 1), *self.config.image_resolution
            ).unflatten(0, (bsize, num_frames))
        img = img * 2.0 - 1.0
        return img.permute(0, 1, 4, 2, 3)

    def camera_keys(self) -> list[str]:
        return list(self.config.image_features)

    def _gather_model_inputs(self, batch: dict[str, Tensor]) -> dict:
        """Pull the frame-stacked tensors out of a training batch."""
        from lerobot.policies.common.vla_utils import pad_vector

        cams = self.camera_keys()
        num_frames = self.config.context_frames
        num_triplets = self.config.num_context_triplets

        obs_images, obs_masks = [], []
        for key in cams:
            stack = self._preprocess_image_stack(batch[key])
            obs_images.append(stack)
            pad_key = f"{key}_is_pad"
            mask = (
                ~batch[pad_key].bool().to(stack.device)
                if pad_key in batch
                else torch.ones(stack.shape[:2], dtype=torch.bool, device=stack.device)
            )
            obs_masks.append(mask)

        device = obs_images[0].device
        states = pad_vector(batch[OBS_STATE].to(device), self.config.max_state_dim)
        if states.ndim == 2:
            states = states[:, None]

        frame_masks = batch.get(OBS_FRAME_MASK)
        frame_masks = (
            frame_masks.bool().to(device)
            if frame_masks is not None
            else torch.ones(states.shape[0], num_frames, dtype=torch.bool, device=device)
        )
        temporal_index = batch.get(OBS_TEMPORAL_INDEX)
        if temporal_index is None:
            stride = float(self.config.chunk_size)
            offsets = torch.arange(num_frames - 1, -1, -1, dtype=torch.float32, device=device) * stride
            temporal_index = offsets[None].expand(states.shape[0], num_frames).contiguous()
        else:
            temporal_index = temporal_index.to(device=device, dtype=torch.float32)

        inputs = {
            "obs_images": obs_images,
            "obs_image_masks": obs_masks,
            "obs_states": states,
            "tokens": batch[OBS_LANGUAGE_TOKENS].to(device),
            "language_masks": batch[OBS_LANGUAGE_ATTENTION_MASK].to(device).bool(),
            "frame_masks": frame_masks,
            "temporal_index": temporal_index,
        }

        if num_triplets > 0 and "A" in self.config.context_mode:
            actions = batch[OBS_CONTEXT_ACTIONS].to(device)
            inputs["context_actions"] = pad_vector(actions, self.config.max_action_dim)
        if num_triplets > 0 and self.config.context_mode == "OAO":
            cons_images, cons_masks = [], []
            for key in cams:
                cons_key = key.replace("observation.images", OBS_CONSEQUENCE_IMAGES)
                stack = self._preprocess_image_stack(batch[cons_key])
                cons_images.append(stack)
                cons_masks.append(
                    torch.ones(stack.shape[:2], dtype=torch.bool, device=stack.device)
                )
            cons_states = pad_vector(
                batch[OBS_CONSEQUENCE_STATE].to(device), self.config.max_state_dim
            )
            inputs["consequence_images"] = cons_images
            inputs["consequence_image_masks"] = cons_masks
            inputs["consequence_states"] = cons_states
        return inputs

    # ------------------------------------------------------------------ training

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        """Flow-matching loss summed over every valid target frame (``L_train``)."""
        from lerobot.policies.common.vla_utils import pad_vector

        inputs = self._gather_model_inputs(batch)
        device = inputs["obs_states"].device

        actions = pad_vector(batch[ACTION].to(device), self.config.max_action_dim)
        if actions.ndim == 3:  # [B, C, A] -> single target frame
            actions = actions[:, None]
        bsize, num_frames = actions.shape[:2]

        noise = self.model.sample_noise(actions.shape, device)
        time = self.model.sample_time(bsize * num_frames, device).view(bsize, num_frames)

        losses = self.model.forward(actions=actions, noise=noise, time=time, **inputs)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[..., :original_action_dim]

        weight = inputs["frame_masks"].to(losses.dtype)[:, :, None, None]
        pad_key = f"{ACTION}_is_pad"
        if pad_key in batch:
            action_pad = batch[pad_key].to(device)
            if action_pad.ndim == 2:
                action_pad = action_pad[:, None]
            weight = weight * (~action_pad.bool()).to(losses.dtype)[..., None]
        weight = weight.expand_as(losses)

        denom = weight.sum().clamp(min=1)
        loss_per_dim = (losses * weight).sum(dim=(0, 1, 2)) / weight.sum(dim=(0, 1, 2)).clamp(min=1)
        loss_dict = {"loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist()}

        if reduction == "none":
            per_sample = (losses * weight).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3)).clamp(min=1)
            loss_dict["loss"] = per_sample.mean().item()
            return per_sample, loss_dict

        loss = (losses * weight).sum() / denom
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    # ----------------------------------------------------------------- inference

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict the current chunk, conditioned on the rolling triplet buffer."""
        from lerobot.policies.common.vla_utils import pad_vector

        self.eval()
        device = next(self.parameters()).device
        cams = self.camera_keys()

        current_images = {key: batch[key].to(device) for key in cams}
        current_state = pad_vector(batch[OBS_STATE].to(device), self.config.max_state_dim)

        # The observation reached now is the consequence of the chunk just executed.
        self._triplet_buffer.close_pending(current_images, current_state)

        context = self._triplet_buffer.context_inputs(
            current_images,
            current_state,
            cams,
            num_frames=self.config.context_frames,
            action_dim=self.config.max_action_dim,
        )
        context["obs_images"] = [self._preprocess_image_stack(x) for x in context["obs_images"]]
        if "consequence_images" in context:
            context["consequence_images"] = [
                self._preprocess_image_stack(x) for x in context["consequence_images"]
            ]

        actions = self.model.sample_actions(
            tokens=batch[OBS_LANGUAGE_TOKENS].to(device),
            language_masks=batch[OBS_LANGUAGE_ATTENTION_MASK].to(device).bool(),
            **context,
            **kwargs,
        )

        self._triplet_buffer.open_pending(current_images, current_state)
        self._triplet_buffer.commit_action(actions)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        self._triplet_buffer.advance(1)
        return self._action_queue.popleft()
