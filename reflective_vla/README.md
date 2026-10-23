# Reflective VLA

Reproduction of **Reflective VLA: In-Context Action Consequences Make VLAs Generalize**
([arXiv:2606.25215](https://arxiv.org/abs/2606.25215), Lian, Yu & Zhang), built as a LeRobot
policy plugin on top of the pi0.5 Mixture-of-Transformers backbone.

The paper's code is not released ("Code Coming Soon"), so this is a from-scratch
implementation from the paper text. See [`PLAN.md`](./PLAN.md) for the full reading of the
paper, the design rationale, and what is and is not implemented yet.

## What it does

A reactive VLA predicts `A_t ~ pi(A_t | L, O_t)`. Embodiment-specific factors — camera
extrinsics, robot calibration, actuation bias — are not identifiable from a single frame, so
the policy memorizes the training embodiments instead of inferring the deployed one.

Reflective VLA conditions each decision on `K-1` completed **observation–action–consequence**
triplets plus the current observation. The consequence `O'_i = O_{i+C}` is what carries the
signal: it exposes how a commanded motion actually translated into observed change, which is
exactly the evidence that identifies the deployment. The paper's ablation isolates this —
`O` only 72.3, `O,A` 73.8, `O,A,O'` **77.8** against a reactive 73.1.

Two mechanisms make it trainable and deployable:

- **Block-causal mask (Eq. 6).** Query slot `k` sees `{T_j : j<k} u {L, tau_k, O_k}` and is
  masked from its own executed action `A_k`, its own consequence `O'_k`, every later triplet,
  and its sibling query slots. One forward pass then supervises all `K` frames with no leakage.
- **Rolling triplet buffer + KV cache.** Completed triplets are encoded once and reused, so a
  prompt 8x longer costs only ~1.4x the latency.

## Layout

```
src/reflective_vla/
  masking.py                      token layout, block-causal mask, RoPE positions
  tokenization.py                 g_A / proprio / tau / boundary projectors, prefix assembly
  configuration_reflective_vla.py paper hyperparameters, Table 4 presets
  modeling_reflective_vla.py      ReflectiveVLAPytorch + ReflectiveVLAPolicy
  buffer.py                       inference-time FIFO triplet buffer
  processor_reflective_vla.py     pre/post-processing pipelines
```

`third_party/lerobot` is vendored and installed editable, but **not patched**: LeRobot
resolves `reflective_vla` through its convention-based plugin lookup.

## Setup

```bash
conda create -n reflective python=3.12 -y && conda activate reflective
pip install -e third_party/lerobot
pip install -e ".[dev]"
pytest -q          # 37 tests, no GPU required
```

## Status

Model, masking and the inference path are implemented and tested. The data pipeline
(triplet sampling with the randomized stride), the training loop, and LIBERO / LIBERO-Plus
evaluation are not built yet — see the milestone table in `PLAN.md`.

## Defaults

Defaults reproduce the paper: `K=8`, `C=10`, PaliGemma-3B prefix, action expert of hidden
width 1024, per-triplet instruction re-insertion, learned frame-boundary tokens, and full
fine-tuning with a `0.1x` learning-rate multiplier on the VLM. Setting `context_frames=1`
gives the paper's matched reactive baseline; `context_mode` switches between the `O` / `OA` /
`OAO` ablation arms.
