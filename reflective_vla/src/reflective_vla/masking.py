# Copyright 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Token layout and block-causal attention masks for Reflective VLA.

Implements the masking rule of *Reflective VLA: In-Context Action Consequences Make
VLAs Generalize* (arXiv:2606.25215).  For target frame ``t_k`` the query slot
``A~_{t_k}`` may only attend to

    V_{t_k} = {T_{t_j} : j < k}  u  {(L, tau_{t_k}, O_{t_k})}          (Eq. 6)

so it is masked from its own executed action ``A_{t_k}``, from its own consequence
``O'_{t_k}``, from every later triplet, and from its sibling query slots.  That lets a
single forward pass supervise all ``K`` sampled frames without leakage.

Everything here is pure ``torch`` with no LeRobot import, so the leakage tests can run
without instantiating a 3B backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

# --- segment kinds ---------------------------------------------------------------
SEG_OBS_IMAGE = 0
SEG_OBS_STATE = 1
SEG_LANGUAGE = 2
SEG_TEMPORAL_INDEX = 3
SEG_ACTION = 4
SEG_CONSEQUENCE_IMAGE = 5
SEG_CONSEQUENCE_STATE = 6
SEG_BOUNDARY = 7

SEGMENT_NAMES = {
    SEG_OBS_IMAGE: "obs_image",
    SEG_OBS_STATE: "obs_state",
    SEG_LANGUAGE: "language",
    SEG_TEMPORAL_INDEX: "temporal_index",
    SEG_ACTION: "action",
    SEG_CONSEQUENCE_IMAGE: "consequence_image",
    SEG_CONSEQUENCE_STATE: "consequence_state",
    SEG_BOUNDARY: "boundary",
}

# --- learned frame-boundary tokens -----------------------------------------------
# The paper's Limitations name "frame-boundary tokens and stride randomization" as the
# mitigation for the shortcut where the policy extrapolates the target chunk from the
# temporal pattern of neighbouring chunks, so these are functional, not decoration.
BOUND_FRAME_SEP = 0
BOUND_FRAME_START = 1
BOUND_ACTION_START = 2
BOUND_ACTION_END = 3
BOUND_FRAME_END = 4
NUM_BOUNDARY_TOKENS = 5

BOUNDARY_NAMES = {
    BOUND_FRAME_SEP: "<|frame_sep|>",
    BOUND_FRAME_START: "<|frame_start|>",
    BOUND_ACTION_START: "<|action_start|>",
    BOUND_ACTION_END: "<|action_end|>",
    BOUND_FRAME_END: "<|frame_end|>",
}

#: Segments that belong to the *evidence* part ``E_k`` of a triplet block.  A frame's own
#: query slot is masked from these; later frames' query slots can read them.  This is the
#: whole point of the architecture: ``A_k`` would leak the answer and ``O'_k`` the future.
EVIDENCE_SEGMENTS = frozenset({SEG_ACTION, SEG_CONSEQUENCE_IMAGE, SEG_CONSEQUENCE_STATE})

CONTEXT_MODES = ("O", "OA", "OAO")


@dataclass(frozen=True)
class Segment:
    """One contiguous run of prefix tokens coming from a single source tensor."""

    kind: int
    frame: int
    length: int
    #: index into the per-frame stack this segment reads from (camera index for images,
    #: boundary-token id for :data:`SEG_BOUNDARY`)
    source_index: int = 0
    #: only meaningful for :data:`SEG_BOUNDARY`, whose side depends on where it sits
    evidence_override: bool | None = None

    @property
    def is_evidence(self) -> bool:
        if self.evidence_override is not None:
            return self.evidence_override
        return self.kind in EVIDENCE_SEGMENTS

    @property
    def name(self) -> str:
        if self.kind == SEG_BOUNDARY:
            return BOUNDARY_NAMES[self.source_index]
        return SEGMENT_NAMES[self.kind]


@dataclass
class PrefixLayout:
    """Ordered prefix segments plus the per-token tags the mask builder needs."""

    segments: list[Segment]
    num_frames: int
    num_cameras: int
    context_mode: str
    block_ids: Tensor = field(repr=False)
    is_evidence: Tensor = field(repr=False)
    segment_kinds: Tensor = field(repr=False)

    @property
    def length(self) -> int:
        return int(self.block_ids.shape[0])

    def to(self, device: torch.device | str) -> PrefixLayout:
        return PrefixLayout(
            segments=self.segments,
            num_frames=self.num_frames,
            num_cameras=self.num_cameras,
            context_mode=self.context_mode,
            block_ids=self.block_ids.to(device),
            is_evidence=self.is_evidence.to(device),
            segment_kinds=self.segment_kinds.to(device),
        )

    def token_slice(self, index: int) -> slice:
        """Token range covered by ``self.segments[index]``."""
        start = sum(seg.length for seg in self.segments[:index])
        return slice(start, start + self.segments[index].length)

    def describe(self) -> str:
        lines, offset = [], 0
        for seg in self.segments:
            tag = "E" if seg.is_evidence else "P"
            lines.append(
                f"  [{offset:5d}:{offset + seg.length:5d}] frame={seg.frame} {tag} {seg.name}"
                + (f"[{seg.source_index}]" if seg.kind in (SEG_OBS_IMAGE, SEG_CONSEQUENCE_IMAGE) else "")
            )
            offset += seg.length
        return f"PrefixLayout(K={self.num_frames}, mode={self.context_mode}, len={offset})\n" + "\n".join(lines)


def build_prefix_layout(
    *,
    num_frames: int,
    num_cameras: int,
    tokens_per_image: int,
    num_language_tokens: int,
    action_history_tokens: int = 8,
    context_mode: str = "OAO",
    language_per_frame: bool = True,
    use_temporal_index: bool = True,
    include_state: bool = True,
    language_first: bool = False,
    use_boundary_tokens: bool = True,
    device: torch.device | str = "cpu",
) -> PrefixLayout:
    """Lay out the ``K`` triplet blocks of the VLM prefix.

    Per-frame order (``language_first=False``, the PaliGemma-native ordering that keeps
    the last block byte-compatible with a pi05 prefix)::

        P_k = [obs images | obs state | language | tau]      <- visible to query k
        E_k = [action | consequence images | consequence state]  <- hidden from query k

    The final frame ``k = K-1`` is the *current* observation and never carries an
    evidence part, matching the paper's template where the last ``<|frame_start|>`` block
    holds only ``OBS_t`` and ``PROPRIO_t``.

    Args:
        num_frames: ``K``, the number of context frames (``K=1`` reduces to reactive pi05).
        num_cameras: image streams per observation (e.g. third-person + wrist).
        tokens_per_image: VLM tokens produced per image (256 for SigLIP @ 224/14).
        num_language_tokens: padded tokenizer length.
        action_history_tokens: tokens produced by the ``g_A`` projection of a past chunk.
        context_mode: ``"O"`` / ``"OA"`` / ``"OAO"`` — the Table 3a ablation axis.
        language_per_frame: re-insert the instruction in every block, as the paper does.
        use_temporal_index: emit the ``tau_i`` token.
        include_state: emit a projected proprioception token (pi05 instead folds a
            discretized state into the prompt; the paper uses a learned projector).
        language_first: put the instruction ahead of the images inside each block.
        use_boundary_tokens: emit the learned ``<|frame_sep|>`` / ``<|frame_start|>`` /
            ``<|action_start|>`` / ``<|action_end|>`` / ``<|frame_end|>`` markers from the
            paper's template.  The Limitations section credits them with mitigating the
            "extrapolate from the past chunk" shortcut, so they default on.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"context_mode must be one of {CONTEXT_MODES}, got {context_mode!r}")
    if num_cameras < 1:
        raise ValueError(f"num_cameras must be >= 1, got {num_cameras}")

    include_action = "A" in context_mode
    include_consequence = context_mode == "OAO"

    def boundary(frame: int, token_id: int, evidence: bool) -> Segment:
        return Segment(SEG_BOUNDARY, frame, 1, source_index=token_id, evidence_override=evidence)

    segments: list[Segment] = []
    for k in range(num_frames):
        language_here = language_per_frame or k == 0
        head: list[Segment] = []
        if use_boundary_tokens:
            if k > 0:
                head.append(boundary(k, BOUND_FRAME_SEP, evidence=False))
            head.append(boundary(k, BOUND_FRAME_START, evidence=False))
        if language_here and language_first:
            head.append(Segment(SEG_LANGUAGE, k, num_language_tokens))
        if use_temporal_index:
            head.append(Segment(SEG_TEMPORAL_INDEX, k, 1))
        for cam in range(num_cameras):
            head.append(Segment(SEG_OBS_IMAGE, k, tokens_per_image, source_index=cam))
        if include_state:
            head.append(Segment(SEG_OBS_STATE, k, 1))
        if language_here and not language_first:
            head.append(Segment(SEG_LANGUAGE, k, num_language_tokens))
        segments.extend(head)

        # The current frame has no executed action and no consequence yet.
        is_context_frame = k < num_frames - 1
        has_evidence = is_context_frame and include_action
        if has_evidence:
            if use_boundary_tokens:
                segments.append(boundary(k, BOUND_ACTION_START, evidence=True))
            segments.append(Segment(SEG_ACTION, k, action_history_tokens))
            if use_boundary_tokens:
                segments.append(boundary(k, BOUND_ACTION_END, evidence=True))
        if is_context_frame and include_consequence:
            for cam in range(num_cameras):
                segments.append(Segment(SEG_CONSEQUENCE_IMAGE, k, tokens_per_image, source_index=cam))
            if include_state:
                segments.append(Segment(SEG_CONSEQUENCE_STATE, k, 1))
        if has_evidence and use_boundary_tokens:
            segments.append(boundary(k, BOUND_FRAME_END, evidence=True))

    block_ids = torch.cat(
        [torch.full((seg.length,), seg.frame, dtype=torch.long) for seg in segments]
    ).to(device)
    is_evidence = torch.cat(
        [torch.full((seg.length,), seg.is_evidence, dtype=torch.bool) for seg in segments]
    ).to(device)
    segment_kinds = torch.cat(
        [torch.full((seg.length,), seg.kind, dtype=torch.long) for seg in segments]
    ).to(device)

    return PrefixLayout(
        segments=segments,
        num_frames=num_frames,
        num_cameras=num_cameras,
        context_mode=context_mode,
        block_ids=block_ids,
        is_evidence=is_evidence,
        segment_kinds=segment_kinds,
    )


def suffix_query_frames(num_frames: int, chunk_size: int, device: torch.device | str = "cpu") -> Tensor:
    """Frame index owning each suffix (action-expert) token: ``[Q_0 x C, ..., Q_{K-1} x C]``."""
    return torch.arange(num_frames, device=device).repeat_interleave(chunk_size)


def visibility_matrix(layout: PrefixLayout, query_frames: Tensor) -> Tensor:
    """``[S, P]`` bool: which prefix tokens each suffix token may read (Eq. 6).

    Query slot ``k`` sees every token of the strictly earlier blocks, plus the
    observable part ``P_k`` of its own block — never its own ``A_k`` or ``O'_k``.
    """
    block_ids = layout.block_ids.to(query_frames.device)
    is_evidence = layout.is_evidence.to(query_frames.device)
    earlier = block_ids[None, :] < query_frames[:, None]
    own_observable = (block_ids[None, :] == query_frames[:, None]) & ~is_evidence[None, :]
    return earlier | own_observable


def build_attention_mask(
    layout: PrefixLayout,
    query_frames: Tensor,
    prefix_pad_masks: Tensor,
    suffix_pad_masks: Tensor,
    *,
    intra_block_causal: bool = False,
) -> Tensor:
    """Full ``[B, T, T]`` boolean attention mask over ``prefix ++ suffix``.

    Rules (row = query token, column = key token):

    ==================  ===========================================================
    prefix -> prefix    ``block[key] <= block[query]``  (block-causal; bidirectional
                        inside a block, which is what makes a completed triplet
                        cacheable as one unit)
    suffix -> prefix    :func:`visibility_matrix`, i.e. Eq. 6
    suffix -> suffix    same query slot only (sibling slots are mutually masked)
    prefix -> suffix    never
    ==================  ===========================================================

    Args:
        intra_block_causal: additionally forbid ``P_k`` from reading ``E_k``.  Not needed
            for the default inference scheme (a triplet is encoded once, complete), but it
            makes ``P_k``'s KV valid *before* the triplet closes, which a streaming cache
            that appends evidence in place would require.
    """
    device = prefix_pad_masks.device
    block_ids = layout.block_ids.to(device)
    is_evidence = layout.is_evidence.to(device)
    query_frames = query_frames.to(device)

    prefix_len = block_ids.shape[0]
    suffix_len = query_frames.shape[0]
    if prefix_pad_masks.shape[1] != prefix_len:
        raise ValueError(
            f"prefix_pad_masks has {prefix_pad_masks.shape[1]} tokens but layout has {prefix_len}"
        )
    if suffix_pad_masks.shape[1] != suffix_len:
        raise ValueError(
            f"suffix_pad_masks has {suffix_pad_masks.shape[1]} tokens but query_frames has {suffix_len}"
        )

    prefix_to_prefix = block_ids[None, :] <= block_ids[:, None]
    if intra_block_causal:
        same_block = block_ids[None, :] == block_ids[:, None]
        evidence_into_observable = same_block & is_evidence[None, :] & ~is_evidence[:, None]
        prefix_to_prefix = prefix_to_prefix & ~evidence_into_observable

    suffix_to_prefix = visibility_matrix(layout, query_frames)
    suffix_to_suffix = query_frames[None, :] == query_frames[:, None]
    prefix_to_suffix = torch.zeros(prefix_len, suffix_len, dtype=torch.bool, device=device)

    att_2d = torch.cat(
        [
            torch.cat([prefix_to_prefix, prefix_to_suffix], dim=1),
            torch.cat([suffix_to_prefix, suffix_to_suffix], dim=1),
        ],
        dim=0,
    )

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    pad_2d = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d[None] & pad_2d


def build_position_ids(
    layout: PrefixLayout,
    query_frames: Tensor,
    prefix_pad_masks: Tensor,
    suffix_pad_masks: Tensor,
) -> Tensor:
    """RoPE positions for ``prefix ++ suffix``.

    The prefix is numbered by ``cumsum(pad) - 1`` exactly as in pi05.  Query slot ``k`` is
    then placed *immediately after the last token it can see*, i.e. at the end of ``P_k``.
    Because everything preceding ``P_k`` is visible to slot ``k``, that offset equals the
    number of valid tokens in ``V_{t_k}`` -- and for ``k = K-1`` it collapses to pi05's
    ``prefix_offsets + cumsum(suffix_pad) - 1``, so training and single-query inference
    put the action tokens at the same relative distance from their own observation.
    """
    device = prefix_pad_masks.device
    prefix_pad = prefix_pad_masks.to(torch.long)
    prefix_positions = torch.cumsum(prefix_pad, dim=1) - 1

    visible = visibility_matrix(layout, query_frames.to(device)).to(torch.long)  # [S, P]
    # [B, S]: how many *valid* prefix tokens each query slot may read.
    # CUDA has no integer matmul ("baddbmm_cuda" not implemented for 'Long'); the counts are
    # far below 2**24 so float32 is exact.
    offsets = torch.einsum("sp,bp->bs", visible.float(), prefix_pad.float()).round().to(torch.long)

    suffix_pad = suffix_pad_masks.to(torch.long)
    query_frames = query_frames.to(device)
    slot_starts = torch.zeros_like(suffix_pad)
    slot_starts[:, 1:] = (query_frames[None, 1:] != query_frames[None, :-1]).to(torch.long)
    slot_group = torch.cumsum(slot_starts, dim=1)
    within_slot = torch.cumsum(suffix_pad, dim=1) - 1
    # Restart the running count at every slot boundary.
    group_base = torch.zeros_like(within_slot)
    for group in range(int(slot_group.max().item()) + 1):
        in_group = slot_group == group
        first = in_group.to(torch.long).argmax(dim=1, keepdim=True)
        group_base = torch.where(in_group, within_slot.gather(1, first)[:, :1], group_base)
    within_slot = within_slot - group_base

    suffix_positions = offsets + within_slot
    return torch.cat([prefix_positions.clamp(min=0), suffix_positions.clamp(min=0)], dim=1)


def build_prefix_attention_mask(
    layout: PrefixLayout,
    prefix_pad_masks: Tensor,
    *,
    intra_block_causal: bool = False,
) -> Tensor:
    """``[B, P, P]`` block-causal mask over the prefix alone (used to prefill the KV cache).

    Block-causality is what makes the cache sound: a completed triplet's keys and values
    never depend on anything that arrives later, so they can be encoded once and reused for
    the rest of the episode.
    """
    device = prefix_pad_masks.device
    block_ids = layout.block_ids.to(device)
    is_evidence = layout.is_evidence.to(device)

    att_2d = block_ids[None, :] <= block_ids[:, None]
    if intra_block_causal:
        same_block = block_ids[None, :] == block_ids[:, None]
        att_2d = att_2d & ~(same_block & is_evidence[None, :] & ~is_evidence[:, None])

    pad_2d = prefix_pad_masks[:, None, :] & prefix_pad_masks[:, :, None]
    return att_2d[None] & pad_2d


def query_frames_for_inference(
    num_frames: int, chunk_size: int, device: torch.device | str = "cpu"
) -> Tensor:
    """At deployment only the current frame is a target, so every suffix token queries it."""
    return torch.full((chunk_size,), num_frames - 1, dtype=torch.long, device=device)
