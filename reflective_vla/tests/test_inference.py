"""Inference-path tests: KV-cache equivalence and the rolling triplet buffer.

The paper's real-time claim rests on encoding past triplets once and reusing their VLM keys
and values.  That is only sound if the cached path computes the same thing as a full joint
forward, which is what :func:`test_cached_denoise_matches_joint_forward` pins down.
"""

from __future__ import annotations

import torch

from lerobot.policies.common.vla_utils import prepare_attention_masks_4d

from reflective_vla.masking import (
    build_attention_mask,
    build_position_ids,
    build_prefix_attention_mask,
    query_frames_for_inference,
)

from .test_model_leakage import ACTION_DIM, BATCH, CHUNK, IMG, make_config, make_inputs, make_model


def test_cached_denoise_matches_joint_forward(tiny_backbone):
    cfg = make_config()
    model = make_model(cfg)
    inputs = make_inputs(cfg)

    prefix_embs, prefix_pads, layout = model.embed_triplet_prefix(**inputs)

    torch.manual_seed(7)
    x_t = torch.randn(BATCH, CHUNK, cfg.max_action_dim)
    timestep = torch.full((BATCH,), 0.3)

    # --- cached path: prefill the prefix once, then denoise against the cache -------
    prefix_att_4d = prepare_attention_masks_4d(build_prefix_attention_mask(layout, prefix_pads))
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_4d,
        position_ids=(torch.cumsum(prefix_pads.long(), dim=1) - 1).clamp(min=0),
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    cached = model.denoise_step(
        prefix_pad_masks=prefix_pads, past_key_values=cache, x_t=x_t, timestep=timestep
    )

    # --- joint path: one forward over prefix ++ the single query slot ---------------
    suffix_embs, suffix_pads, adarms = model.embed_query_slots(x_t[:, None], timestep[:, None])
    query_frames = query_frames_for_inference(cfg.context_frames, cfg.chunk_size)
    att_4d = prepare_attention_masks_4d(
        build_attention_mask(layout, query_frames, prefix_pads, suffix_pads)
    )
    positions = build_position_ids(layout, query_frames, prefix_pads, suffix_pads)
    (_, suffix_out), _ = model.paligemma_with_expert.forward(
        attention_mask=att_4d,
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms],
    )
    joint = model.action_out_proj(suffix_out.to(torch.float32))

    assert torch.allclose(cached, joint, atol=1e-5), (
        f"cached and joint denoising diverge by {(cached - joint).abs().max():.2e}"
    )


def test_inference_query_sees_the_whole_valid_prefix(tiny_backbone):
    """The current block has no evidence, so at deployment Eq. 6 collapses to pi0.5's rule."""
    cfg = make_config()
    model = make_model(cfg)
    inputs = make_inputs(cfg)
    _, prefix_pads, layout = model.embed_triplet_prefix(**inputs)
    query_frames = query_frames_for_inference(cfg.context_frames, cfg.chunk_size)
    suffix_pads = torch.ones(BATCH, cfg.chunk_size, dtype=torch.bool)
    mask = build_attention_mask(layout, query_frames, prefix_pads, suffix_pads)
    suffix_to_prefix = mask[:, layout.length :, : layout.length]
    assert torch.equal(
        suffix_to_prefix, prefix_pads[:, None, :].expand_as(suffix_to_prefix)
    )


# --------------------------------------------------------------------------- rollout


def make_policy(cfg):
    from reflective_vla.modeling_reflective_vla import ReflectiveVLAPolicy

    torch.manual_seed(0)
    return ReflectiveVLAPolicy(cfg).eval()


def make_observation(cfg, seed: int):
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    torch.manual_seed(seed)
    cams = list(cfg.image_features)
    batch = {cam: torch.rand(BATCH, 3, IMG, IMG) for cam in cams}
    batch[OBS_STATE] = torch.randn(BATCH, cfg.output_features["action"].shape[0])
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(0, 60, (BATCH, cfg.tokenizer_max_length))
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(
        BATCH, cfg.tokenizer_max_length, dtype=torch.bool
    )
    return batch


def test_buffer_fills_over_a_rollout(tiny_backbone):
    cfg = make_config()
    policy = make_policy(cfg)
    policy.reset()

    # A triplet only closes once its consequence has been observed, which happens on the
    # first step of the *next* chunk -- so the count lags one chunk behind.
    expected_after_chunk = [0, 1, 2, 3, 3]  # capacity K-1 = 3, then FIFO eviction holds
    for chunk_index in range(5):
        batch = make_observation(cfg, seed=chunk_index)
        for _ in range(cfg.n_action_steps):
            action = policy.select_action(batch)
            assert action.shape == (BATCH, ACTION_DIM)
        assert len(policy._triplet_buffer) == expected_after_chunk[chunk_index]
    assert len(policy._triplet_buffer) == cfg.num_context_triplets


def test_early_episode_context_is_masked_not_faked(tiny_backbone):
    """Before K-1 triplets exist the empty slots are masked, exactly as in training."""
    cfg = make_config()
    policy = make_policy(cfg)
    policy.reset()

    batch = make_observation(cfg, seed=0)
    policy.predict_action_chunk(batch)
    context = policy._triplet_buffer.context_inputs(
        {cam: batch[cam] for cam in cfg.image_features},
        torch.randn(BATCH, cfg.max_state_dim),
        list(cfg.image_features),
        num_frames=cfg.context_frames,
        action_dim=cfg.max_action_dim,
    )
    assert context["frame_masks"][0].tolist() == [False, False, False, True]


def test_temporal_index_tracks_the_executed_stride(tiny_backbone):
    cfg = make_config()
    policy = make_policy(cfg)
    policy.reset()

    for chunk_index in range(3):
        batch = make_observation(cfg, seed=chunk_index)
        for _ in range(cfg.n_action_steps):
            policy.select_action(batch)

    context = policy._triplet_buffer.context_inputs(
        {cam: batch[cam] for cam in cfg.image_features},
        torch.randn(BATCH, cfg.max_state_dim),
        list(cfg.image_features),
        num_frames=cfg.context_frames,
        action_dim=cfg.max_action_dim,
    )
    tau = context["temporal_index"][0].tolist()
    assert tau[-1] == 0.0
    filled = [t for t, m in zip(tau, context["frame_masks"][0].tolist(), strict=True) if m]
    # Offsets grow backwards in multiples of the chunk horizon.
    assert filled[:-1] == sorted(filled[:-1], reverse=True)
    assert all(t % cfg.chunk_size == 0 for t in filled)


def test_reset_clears_context_between_episodes(tiny_backbone):
    cfg = make_config()
    policy = make_policy(cfg)
    policy.reset()
    batch = make_observation(cfg, seed=0)
    for _ in range(2 * cfg.n_action_steps):
        policy.select_action(batch)
    assert len(policy._triplet_buffer) > 0
    policy.reset()
    assert len(policy._triplet_buffer) == 0
    assert policy._triplet_buffer.step == 0
