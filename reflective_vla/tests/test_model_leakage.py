"""End-to-end leakage tests through the real assembly + mask + attention path.

If a frame's query slot could see its own executed action or its own consequence, the
flow-matching target would be trivially recoverable and every reported number would be
meaningless. These tests perturb one piece of evidence at a time and assert exactly which
per-frame losses may move.
"""

from __future__ import annotations

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from reflective_vla.configuration_reflective_vla import ReflectiveVLAConfig

K = 4
CHUNK = 3
BATCH = 2
STATE_DIM = 7
ACTION_DIM = 7
IMG = 8
CAMS = ["observation.images.top", "observation.images.wrist"]


def make_config(**overrides) -> ReflectiveVLAConfig:
    kwargs = {
        "context_frames": K,
        "chunk_size": CHUNK,
        "n_action_steps": CHUNK,
        "max_state_dim": STATE_DIM,
        "max_action_dim": ACTION_DIM,
        "image_resolution": (IMG, IMG),
        "tokenizer_max_length": 5,
        "dtype": "float32",
        "device": "cpu",
        "num_inference_steps": 2,
    }
    kwargs.update(overrides)
    cfg = ReflectiveVLAConfig(**kwargs)
    cfg.input_features = {
        cam: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG, IMG)) for cam in CAMS
    }
    cfg.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    return cfg


def make_model(cfg):
    from reflective_vla.modeling_reflective_vla import ReflectiveVLAPytorch

    torch.manual_seed(0)
    model = ReflectiveVLAPytorch(cfg).eval()
    return model


def make_inputs(cfg, seed: int = 0):
    torch.manual_seed(seed)
    n_frames = cfg.context_frames
    n_trip = cfg.num_context_triplets
    ones = torch.ones(BATCH, n_frames, dtype=torch.bool)
    trip_ones = torch.ones(BATCH, max(n_trip, 1), dtype=torch.bool)
    inputs = {
        "obs_images": [torch.randn(BATCH, n_frames, 3, IMG, IMG) for _ in CAMS],
        "obs_image_masks": [ones.clone() for _ in CAMS],
        "obs_states": torch.randn(BATCH, n_frames, STATE_DIM),
        "tokens": torch.randint(0, 60, (BATCH, cfg.tokenizer_max_length)),
        "language_masks": torch.ones(BATCH, cfg.tokenizer_max_length, dtype=torch.bool),
        "frame_masks": ones.clone(),
        "temporal_index": torch.arange(n_frames - 1, -1, -1).float()[None].expand(BATCH, n_frames)
        * cfg.chunk_size,
    }
    if n_trip > 0:
        inputs["consequence_images"] = [torch.randn(BATCH, n_trip, 3, IMG, IMG) for _ in CAMS]
        inputs["consequence_image_masks"] = [trip_ones.clone() for _ in CAMS]
        inputs["consequence_states"] = torch.randn(BATCH, n_trip, STATE_DIM)
        inputs["context_actions"] = torch.randn(BATCH, n_trip, cfg.chunk_size, ACTION_DIM)
    return inputs


def per_frame_loss(model, cfg, inputs):
    torch.manual_seed(123)
    actions = torch.randn(BATCH, cfg.context_frames, cfg.chunk_size, ACTION_DIM)
    noise = torch.randn_like(actions)
    time = torch.full((BATCH, cfg.context_frames), 0.4)
    with torch.no_grad():
        losses = model.forward(actions=actions, noise=noise, time=time, **inputs)
    return losses.mean(dim=(0, 2, 3))  # [K]


def test_forward_shapes_and_finiteness(tiny_backbone):
    cfg = make_config()
    model = make_model(cfg)
    inputs = make_inputs(cfg)
    torch.manual_seed(123)
    actions = torch.randn(BATCH, K, CHUNK, ACTION_DIM)
    losses = model.forward(
        actions=actions,
        noise=torch.randn_like(actions),
        time=torch.full((BATCH, K), 0.4),
        **inputs,
    )
    assert losses.shape == (BATCH, K, CHUNK, ACTION_DIM)
    assert torch.isfinite(losses).all()


def test_consequence_of_frame_k_is_invisible_to_frame_k(tiny_backbone):
    """O'_k must not reach query k, but must reach every later query."""
    cfg = make_config()
    model = make_model(cfg)
    inputs = make_inputs(cfg)
    base = per_frame_loss(model, cfg, inputs)

    for k in range(cfg.num_context_triplets):
        perturbed = dict(inputs)
        perturbed["consequence_images"] = [img.clone() for img in inputs["consequence_images"]]
        perturbed["consequence_images"][0][:, k] += 5.0
        after = per_frame_loss(model, cfg, perturbed)

        assert torch.allclose(base[k], after[k], atol=1e-6), f"O'_{k} leaked into query {k}"
        assert not torch.allclose(base[k + 1], after[k + 1], atol=1e-6), (
            f"O'_{k} did not reach query {k + 1}"
        )
        for j in range(k):
            assert torch.allclose(base[j], after[j], atol=1e-6), f"O'_{k} reached earlier query {j}"


def test_executed_action_of_frame_k_is_invisible_to_frame_k(tiny_backbone):
    """A_k is the answer for query k; it may only be read by later queries."""
    cfg = make_config()
    model = make_model(cfg)
    inputs = make_inputs(cfg)
    base = per_frame_loss(model, cfg, inputs)

    for k in range(cfg.num_context_triplets):
        perturbed = dict(inputs)
        perturbed["context_actions"] = inputs["context_actions"].clone()
        perturbed["context_actions"][:, k] += 5.0
        after = per_frame_loss(model, cfg, perturbed)

        assert torch.allclose(base[k], after[k], atol=1e-6), f"A_{k} leaked into query {k}"
        assert not torch.allclose(base[k + 1], after[k + 1], atol=1e-6), (
            f"A_{k} did not reach query {k + 1}"
        )


def test_observation_of_frame_k_reaches_frame_k_and_later_only(tiny_backbone):
    cfg = make_config()
    model = make_model(cfg)
    inputs = make_inputs(cfg)
    base = per_frame_loss(model, cfg, inputs)

    k = 1
    perturbed = dict(inputs)
    perturbed["obs_images"] = [img.clone() for img in inputs["obs_images"]]
    perturbed["obs_images"][0][:, k] += 5.0
    after = per_frame_loss(model, cfg, perturbed)

    for j in range(k):
        assert torch.allclose(base[j], after[j], atol=1e-6), f"O_{k} reached earlier query {j}"
    for j in range(k, cfg.context_frames):
        assert not torch.allclose(base[j], after[j], atol=1e-6), f"O_{k} did not reach query {j}"


def test_future_frames_never_influence_earlier_queries(tiny_backbone):
    cfg = make_config()
    model = make_model(cfg)
    inputs = make_inputs(cfg)
    base = per_frame_loss(model, cfg, inputs)

    last = cfg.context_frames - 1
    perturbed = dict(inputs)
    perturbed["obs_states"] = inputs["obs_states"].clone()
    perturbed["obs_states"][:, last] += 5.0
    after = per_frame_loss(model, cfg, perturbed)
    assert torch.allclose(base[:last], after[:last], atol=1e-6)
    assert not torch.allclose(base[last], after[last], atol=1e-6)


def test_history_only_ablation_drops_consequence(tiny_backbone):
    """context_mode='OA' is the paper's matched history-only baseline."""
    cfg = make_config(context_mode="OA")
    model = make_model(cfg)
    inputs = make_inputs(cfg)
    inputs.pop("consequence_images")
    inputs.pop("consequence_image_masks")
    inputs.pop("consequence_states")
    losses = per_frame_loss(model, cfg, inputs)
    assert torch.isfinite(losses).all()


def test_reactive_k1_runs(tiny_backbone):
    cfg = make_config(context_frames=1)
    assert cfg.is_reactive
    model = make_model(cfg)
    inputs = make_inputs(cfg)
    losses = per_frame_loss(model, cfg, inputs)
    assert losses.shape == (1,)
    assert torch.isfinite(losses).all()


def test_masked_frames_are_ignored_by_later_queries(tiny_backbone):
    """A slot the episode could not fill must behave as if it were not there."""
    cfg = make_config()
    model = make_model(cfg)
    inputs = make_inputs(cfg)
    inputs["frame_masks"] = inputs["frame_masks"].clone()
    inputs["frame_masks"][:, 0] = False
    base = per_frame_loss(model, cfg, inputs)

    perturbed = dict(inputs)
    perturbed["obs_images"] = [img.clone() for img in inputs["obs_images"]]
    perturbed["obs_images"][0][:, 0] += 5.0
    perturbed["consequence_images"] = [img.clone() for img in inputs["consequence_images"]]
    perturbed["consequence_images"][0][:, 0] += 5.0
    after = per_frame_loss(model, cfg, perturbed)
    assert torch.allclose(base[1:], after[1:], atol=1e-6)


@pytest.mark.parametrize("num_frames", [1, 4])
def test_sample_actions_shape(tiny_backbone, num_frames):
    cfg = make_config(context_frames=num_frames)
    model = make_model(cfg)
    inputs = make_inputs(cfg)
    actions = model.sample_actions(**inputs)
    assert actions.shape == (BATCH, CHUNK, ACTION_DIM)
    assert torch.isfinite(actions).all()
