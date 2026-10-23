"""Leakage and layout tests for the Reflective VLA block-causal mask (Eq. 6).

These are the load-bearing tests of the whole implementation: if a query slot can see
its own executed action or its own consequence, the model trains on a leaked target and
every downstream number is meaningless.
"""

from __future__ import annotations

import pytest
import torch

from reflective_vla.masking import (
    SEG_ACTION,
    SEG_BOUNDARY,
    SEG_CONSEQUENCE_IMAGE,
    SEG_CONSEQUENCE_STATE,
    SEG_OBS_IMAGE,
    build_attention_mask,
    build_position_ids,
    build_prefix_layout,
    suffix_query_frames,
)

TOKENS_PER_IMAGE = 4
NUM_LANG = 3
ACTION_TOKENS = 8
NUM_CAMERAS = 2


def make_layout(num_frames: int, context_mode: str = "OAO", **kwargs):
    return build_prefix_layout(
        num_frames=num_frames,
        num_cameras=NUM_CAMERAS,
        tokens_per_image=TOKENS_PER_IMAGE,
        num_language_tokens=NUM_LANG,
        action_history_tokens=ACTION_TOKENS,
        context_mode=context_mode,
        **kwargs,
    )


def full_masks(layout, chunk_size, batch_size=2, **kwargs):
    query_frames = suffix_query_frames(layout.num_frames, chunk_size)
    prefix_pad = torch.ones(batch_size, layout.length, dtype=torch.bool)
    suffix_pad = torch.ones(batch_size, query_frames.shape[0], dtype=torch.bool)
    mask = build_attention_mask(layout, query_frames, prefix_pad, suffix_pad, **kwargs)
    return query_frames, prefix_pad, suffix_pad, mask


# --------------------------------------------------------------------------------- layout


def test_last_frame_has_no_evidence():
    """The current observation block carries no executed action and no consequence."""
    layout = make_layout(num_frames=4)
    last = layout.num_frames - 1
    assert not any(seg.is_evidence for seg in layout.segments if seg.frame == last)
    assert any(seg.is_evidence for seg in layout.segments if seg.frame == 0)


def test_context_mode_controls_evidence():
    assert not make_layout(3, "O").is_evidence.any()

    oa = make_layout(3, "OA")
    kinds = {seg.kind for seg in oa.segments if seg.is_evidence}
    assert SEG_ACTION in kinds
    assert SEG_CONSEQUENCE_IMAGE not in kinds

    oao = make_layout(3, "OAO")
    kinds = {seg.kind for seg in oao.segments if seg.is_evidence}
    assert {SEG_ACTION, SEG_CONSEQUENCE_IMAGE, SEG_CONSEQUENCE_STATE} <= kinds


def test_layout_length_matches_segments():
    layout = make_layout(5)
    assert layout.length == sum(seg.length for seg in layout.segments)
    assert layout.block_ids.shape == layout.is_evidence.shape == layout.segment_kinds.shape


def test_boundary_tokens_can_be_disabled():
    with_boundaries = make_layout(3, use_boundary_tokens=True)
    without = make_layout(3, use_boundary_tokens=False)
    assert (with_boundaries.segment_kinds == SEG_BOUNDARY).sum() > 0
    assert (without.segment_kinds == SEG_BOUNDARY).sum() == 0
    assert with_boundaries.length > without.length


# ------------------------------------------------------------------------------- leakage


@pytest.mark.parametrize("num_frames", [2, 3, 8])
def test_query_cannot_see_own_evidence(num_frames):
    """Eq. 6: query k is masked from its own A_k and its own O'_k."""
    layout = make_layout(num_frames)
    chunk = 3
    query_frames, _, _, mask = full_masks(layout, chunk)
    prefix_len = layout.length
    suffix_rows = mask[0, prefix_len:, :prefix_len]

    for row, k in enumerate(query_frames.tolist()):
        own_evidence = (layout.block_ids == k) & layout.is_evidence
        assert not suffix_rows[row][own_evidence].any(), f"query slot {k} leaked its own evidence"


@pytest.mark.parametrize("num_frames", [2, 3, 8])
def test_query_sees_all_earlier_triplets_and_own_observation(num_frames):
    layout = make_layout(num_frames)
    chunk = 3
    query_frames, _, _, mask = full_masks(layout, chunk)
    prefix_len = layout.length
    suffix_rows = mask[0, prefix_len:, :prefix_len]

    for row, k in enumerate(query_frames.tolist()):
        earlier = layout.block_ids < k
        assert suffix_rows[row][earlier].all(), f"query slot {k} cannot see triplet j<k"

        own_observable = (layout.block_ids == k) & ~layout.is_evidence
        assert suffix_rows[row][own_observable].all(), f"query slot {k} cannot see its own O_k"

        later = layout.block_ids > k
        assert not suffix_rows[row][later].any(), f"query slot {k} sees a future triplet"


def test_sibling_query_slots_are_mutually_masked():
    layout = make_layout(4)
    chunk = 3
    query_frames, _, _, mask = full_masks(layout, chunk)
    prefix_len = layout.length
    suffix_block = mask[0, prefix_len:, prefix_len:]
    same_slot = query_frames[None, :] == query_frames[:, None]
    assert torch.equal(suffix_block, same_slot)


def test_prefix_never_attends_to_suffix():
    layout = make_layout(3)
    query_frames, _, _, mask = full_masks(layout, 3)
    prefix_len = layout.length
    assert not mask[:, :prefix_len, prefix_len:].any()


def test_prefix_is_block_causal():
    layout = make_layout(4)
    _, _, _, mask = full_masks(layout, 3)
    prefix_len = layout.length
    prefix_block = mask[0, :prefix_len, :prefix_len]
    expected = layout.block_ids[None, :] <= layout.block_ids[:, None]
    assert torch.equal(prefix_block, expected)


def test_intra_block_causal_hides_evidence_from_observation():
    layout = make_layout(3)
    query_frames = suffix_query_frames(layout.num_frames, 2)
    prefix_pad = torch.ones(1, layout.length, dtype=torch.bool)
    suffix_pad = torch.ones(1, query_frames.shape[0], dtype=torch.bool)

    default = build_attention_mask(layout, query_frames, prefix_pad, suffix_pad)
    causal = build_attention_mask(
        layout, query_frames, prefix_pad, suffix_pad, intra_block_causal=True
    )
    p = layout.length
    obs_rows = (~layout.is_evidence).nonzero(as_tuple=True)[0]
    ev_cols = layout.is_evidence.nonzero(as_tuple=True)[0]

    # Same block, observation row, evidence column: allowed by default, blocked with the flag.
    same_block = layout.block_ids[obs_rows][:, None] == layout.block_ids[ev_cols][None, :]
    d = default[0, :p, :p][obs_rows][:, ev_cols]
    c = causal[0, :p, :p][obs_rows][:, ev_cols]
    assert (d & same_block).any()
    assert not (c & same_block).any()
    # Suffix rows are untouched by the flag.
    assert torch.equal(default[0, p:, :], causal[0, p:, :])


def test_padding_is_masked_both_ways():
    layout = make_layout(3)
    chunk = 2
    query_frames = suffix_query_frames(layout.num_frames, chunk)
    prefix_pad = torch.ones(1, layout.length, dtype=torch.bool)
    prefix_pad[0, 5:9] = False
    suffix_pad = torch.ones(1, query_frames.shape[0], dtype=torch.bool)
    mask = build_attention_mask(layout, query_frames, prefix_pad, suffix_pad)
    assert not mask[0, :, 5:9].any()
    assert not mask[0, 5:9, :].any()


# ------------------------------------------------------------------- reactive (K=1) parity


def test_k1_reduces_to_reactive_pi05():
    """With K=1 there is no evidence anywhere, so the suffix sees the whole prefix."""
    layout = make_layout(1)
    assert not layout.is_evidence.any()
    chunk = 5
    query_frames, prefix_pad, suffix_pad, mask = full_masks(layout, chunk)
    prefix_len = layout.length

    assert mask[0, :prefix_len, :prefix_len].all()  # fully bidirectional prefix
    assert mask[0, prefix_len:, :prefix_len].all()  # suffix reads all of it
    assert mask[0, prefix_len:, prefix_len:].all()  # single slot, bidirectional

    # pi05 numbers the suffix as prefix_offset + cumsum(suffix_pad) - 1.
    positions = build_position_ids(layout, query_frames, prefix_pad, suffix_pad)
    expected_suffix = prefix_pad.sum(dim=1, keepdim=True) + torch.arange(chunk)[None, :]
    assert torch.equal(positions[:, prefix_len:], expected_suffix)


# ----------------------------------------------------------------------------- positions


def test_prefix_positions_match_pi05_cumsum():
    layout = make_layout(3)
    chunk = 2
    query_frames = suffix_query_frames(layout.num_frames, chunk)
    prefix_pad = torch.ones(2, layout.length, dtype=torch.bool)
    prefix_pad[1, 4:7] = False
    suffix_pad = torch.ones(2, query_frames.shape[0], dtype=torch.bool)
    positions = build_position_ids(layout, query_frames, prefix_pad, suffix_pad)
    expected = (torch.cumsum(prefix_pad.long(), dim=1) - 1).clamp(min=0)
    assert torch.equal(positions[:, : layout.length], expected)


def test_query_slot_sits_right_after_its_own_observation():
    """Slot k starts at |V_{t_k}|, i.e. immediately after the last token it can read."""
    layout = make_layout(4)
    chunk = 3
    query_frames = suffix_query_frames(layout.num_frames, chunk)
    prefix_pad = torch.ones(1, layout.length, dtype=torch.bool)
    suffix_pad = torch.ones(1, query_frames.shape[0], dtype=torch.bool)
    positions = build_position_ids(layout, query_frames, prefix_pad, suffix_pad)
    mask = build_attention_mask(layout, query_frames, prefix_pad, suffix_pad)

    prefix_len = layout.length
    for slot in range(layout.num_frames):
        row = slot * chunk
        visible_count = int(mask[0, prefix_len + row, :prefix_len].sum())
        assert int(positions[0, prefix_len + row]) == visible_count
        # positions advance by one within the slot
        got = positions[0, prefix_len + row : prefix_len + row + chunk]
        assert torch.equal(got, visible_count + torch.arange(chunk))


def test_query_positions_are_frame_local_not_global():
    """Slot k must not be pushed past later triplets it cannot see."""
    layout = make_layout(4)
    chunk = 3
    query_frames = suffix_query_frames(layout.num_frames, chunk)
    prefix_pad = torch.ones(1, layout.length, dtype=torch.bool)
    suffix_pad = torch.ones(1, query_frames.shape[0], dtype=torch.bool)
    positions = build_position_ids(layout, query_frames, prefix_pad, suffix_pad)
    starts = [int(positions[0, layout.length + s * chunk]) for s in range(layout.num_frames)]
    assert starts == sorted(starts)
    assert starts[0] < layout.length  # early slots stay near their own frame
    assert starts[-1] == layout.length


def test_suffix_padding_does_not_shift_later_slots():
    layout = make_layout(3)
    chunk = 4
    query_frames = suffix_query_frames(layout.num_frames, chunk)
    prefix_pad = torch.ones(1, layout.length, dtype=torch.bool)
    suffix_pad = torch.ones(1, query_frames.shape[0], dtype=torch.bool)
    suffix_pad[0, 2:4] = False  # tail of slot 0 is padded
    positions = build_position_ids(layout, query_frames, prefix_pad, suffix_pad)
    slot1_start = int(positions[0, layout.length + chunk])
    visible = build_attention_mask(layout, query_frames, prefix_pad, suffix_pad)
    assert slot1_start == int(visible[0, layout.length + chunk, : layout.length].sum())


def test_image_segments_cover_every_camera_and_frame():
    layout = make_layout(3)
    obs = [seg for seg in layout.segments if seg.kind == SEG_OBS_IMAGE]
    assert len(obs) == 3 * NUM_CAMERAS
    cons = [seg for seg in layout.segments if seg.kind == SEG_CONSEQUENCE_IMAGE]
    assert len(cons) == 2 * NUM_CAMERAS  # only the K-1 context frames
