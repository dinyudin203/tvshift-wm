# Copyright 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Projectors and prefix assembly for Reflective VLA.

The paper routes every modality into one VLM token sequence: images go through the native
visual pathway, while "proprioceptive states and historical action chunks are projected into
the token space by lightweight two-layer nonlinear fully connected projectors", with "each
historical action chunk represented by eight learned tokens".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .masking import (
    NUM_BOUNDARY_TOKENS,
    SEG_ACTION,
    SEG_BOUNDARY,
    SEG_CONSEQUENCE_IMAGE,
    SEG_CONSEQUENCE_STATE,
    SEG_LANGUAGE,
    SEG_OBS_IMAGE,
    SEG_OBS_STATE,
    SEG_TEMPORAL_INDEX,
    PrefixLayout,
)


class TwoLayerProjector(nn.Module):
    """The paper's "lightweight two-layer nonlinear fully connected projector"."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class TripletProjectors(nn.Module):
    """Everything Reflective VLA adds on top of a pi0.5 prefix.

    ``g_A`` maps a whole executed chunk ``A_i in R^{C x action_dim}`` to
    ``action_history_tokens`` VLM tokens in one shot, which is what "each historical action
    chunk is represented by eight learned tokens" describes.
    """

    def __init__(
        self,
        *,
        width: int,
        max_state_dim: int,
        max_action_dim: int,
        chunk_size: int,
        action_history_tokens: int = 8,
        use_boundary_tokens: bool = True,
        use_temporal_index: bool = True,
        state_as_token: bool = True,
        temporal_min_period: float = 4e-3,
        temporal_max_period: float = 4.0,
    ):
        super().__init__()
        self.width = width
        self.action_history_tokens = action_history_tokens
        self.chunk_size = chunk_size
        self.max_action_dim = max_action_dim
        self.temporal_min_period = temporal_min_period
        self.temporal_max_period = temporal_max_period

        self.state_proj = (
            TwoLayerProjector(max_state_dim, width, width) if state_as_token else None
        )
        self.action_history_proj = TwoLayerProjector(
            chunk_size * max_action_dim, width, action_history_tokens * width
        )
        self.temporal_index_proj = (
            TwoLayerProjector(width, width, width) if use_temporal_index else None
        )
        self.boundary_embed = (
            nn.Embedding(NUM_BOUNDARY_TOKENS, width) if use_boundary_tokens else None
        )

    def embed_state(self, states: Tensor) -> Tensor:
        """``[B, F, state_dim] -> [B, F, 1, width]``."""
        if self.state_proj is None:
            raise RuntimeError("state_as_token is disabled, no state projector was built")
        return self.state_proj(states).unsqueeze(-2)

    def embed_action_history(self, actions: Tensor) -> Tensor:
        """``[B, K-1, C, action_dim] -> [B, K-1, action_history_tokens, width]`` (``g_A``)."""
        bsize, num_triplets = actions.shape[:2]
        flat = actions.reshape(bsize, num_triplets, -1)
        out = self.action_history_proj(flat)
        return out.view(bsize, num_triplets, self.action_history_tokens, self.width)

    def embed_temporal_index(self, temporal_index: Tensor) -> Tensor:
        """``[B, K] -> [B, K, 1, width]``.

        ``temporal_index`` is the backward offset in environment steps from the query frame,
        so the model can tell a randomized stride apart from a fixed one.
        """
        if self.temporal_index_proj is None:
            raise RuntimeError("use_temporal_index is disabled, no temporal projector was built")
        from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding

        emb = create_sinusoidal_pos_embedding(
            temporal_index.to(torch.float32),
            self.width,
            min_period=self.temporal_min_period,
            max_period=self.temporal_max_period,
            device=temporal_index.device,
        )
        emb = emb.to(dtype=self.temporal_index_proj.fc1.weight.dtype)
        return self.temporal_index_proj(emb).unsqueeze(-2)

    def embed_boundaries(self) -> Tensor:
        """``[NUM_BOUNDARY_TOKENS, width]``."""
        if self.boundary_embed is None:
            raise RuntimeError("use_boundary_tokens is disabled, no boundary embedding was built")
        return self.boundary_embed.weight


@dataclass
class PrefixSources:
    """Per-modality embeddings, already stacked along a frame axis.

    Observation stacks are length ``K``; consequence and action stacks are length ``K-1``,
    since the current frame has neither an executed action nor a consequence yet.
    """

    #: per camera, ``[B, K, tokens_per_image, width]``
    obs_images: list[Tensor]
    #: per camera, ``[B, K]`` -- False for an absent camera
    obs_image_masks: list[Tensor]
    #: ``[B, K]`` -- False for a triplet slot the episode could not fill yet
    frame_masks: Tensor
    #: ``[B, num_language_tokens, width]`` shared by every block (the instruction is constant)
    language: Tensor | None = None
    #: ``[B, num_language_tokens]``
    language_mask: Tensor | None = None
    #: ``[B, K, 1, width]``
    obs_state: Tensor | None = None
    #: per camera, ``[B, K-1, tokens_per_image, width]``
    consequence_images: list[Tensor] = field(default_factory=list)
    #: per camera, ``[B, K-1]``
    consequence_image_masks: list[Tensor] = field(default_factory=list)
    #: ``[B, K-1, 1, width]``
    consequence_state: Tensor | None = None
    #: ``[B, K-1, action_history_tokens, width]``
    action_history: Tensor | None = None
    #: ``[B, K, 1, width]``
    temporal_index: Tensor | None = None
    #: ``[NUM_BOUNDARY_TOKENS, width]``
    boundaries: Tensor | None = None


def assemble_prefix(layout: PrefixLayout, sources: PrefixSources) -> tuple[Tensor, Tensor]:
    """Materialize the prefix in layout order.

    Returns ``(embeddings [B, P, width], pad_masks [B, P])``.  A frame whose
    ``frame_masks`` entry is False has its entire block padded out, which is how both an
    episode that has not produced ``K-1`` triplets yet and an inference buffer that is still
    filling are represented -- training and deployment then see the same thing.
    """
    frame_masks = sources.frame_masks
    bsize = frame_masks.shape[0]
    device = frame_masks.device

    embs: list[Tensor] = []
    pads: list[Tensor] = []

    def push(emb: Tensor, pad: Tensor) -> None:
        embs.append(emb)
        pads.append(pad)

    for seg in layout.segments:
        k = seg.frame
        alive = frame_masks[:, k]  # [B]

        if seg.kind == SEG_OBS_IMAGE:
            emb = sources.obs_images[seg.source_index][:, k]
            valid = sources.obs_image_masks[seg.source_index][:, k] & alive
            push(emb, valid[:, None].expand(bsize, emb.shape[1]))

        elif seg.kind == SEG_CONSEQUENCE_IMAGE:
            emb = sources.consequence_images[seg.source_index][:, k]
            valid = sources.consequence_image_masks[seg.source_index][:, k] & alive
            push(emb, valid[:, None].expand(bsize, emb.shape[1]))

        elif seg.kind == SEG_OBS_STATE:
            emb = sources.obs_state[:, k]
            push(emb, alive[:, None].expand(bsize, emb.shape[1]))

        elif seg.kind == SEG_CONSEQUENCE_STATE:
            emb = sources.consequence_state[:, k]
            push(emb, alive[:, None].expand(bsize, emb.shape[1]))

        elif seg.kind == SEG_ACTION:
            emb = sources.action_history[:, k]
            push(emb, alive[:, None].expand(bsize, emb.shape[1]))

        elif seg.kind == SEG_TEMPORAL_INDEX:
            emb = sources.temporal_index[:, k]
            push(emb, alive[:, None].expand(bsize, emb.shape[1]))

        elif seg.kind == SEG_LANGUAGE:
            emb = sources.language
            push(emb, sources.language_mask & alive[:, None])

        elif seg.kind == SEG_BOUNDARY:
            emb = sources.boundaries[seg.source_index][None, None].expand(bsize, 1, -1)
            push(emb, alive[:, None])

        else:  # pragma: no cover - guarded by the layout builder
            raise ValueError(f"unhandled segment kind {seg.kind}")

    prefix_embs = torch.cat(embs, dim=1)
    prefix_pads = torch.cat(pads, dim=1).to(device=device, dtype=torch.bool)
    if prefix_embs.shape[1] != layout.length:
        raise RuntimeError(
            f"assembled prefix has {prefix_embs.shape[1]} tokens, layout expects {layout.length}"
        )
    return prefix_embs, prefix_pads


def pool_image_tokens(tokens: Tensor, factor: int) -> Tensor:
    """Average-pool along the token axis. ``factor=1`` is the paper's behaviour (no pooling)."""
    if factor <= 1:
        return tokens
    *lead, num_tokens, width = tokens.shape
    if num_tokens % factor != 0:
        raise ValueError(f"cannot pool {num_tokens} tokens by {factor}")
    return tokens.reshape(*lead, num_tokens // factor, factor, width).mean(dim=-2)
