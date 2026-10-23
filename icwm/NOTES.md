# ICWM: In-Context World Modeling for Robotic Control

Implementation of "In-Context World Modeling for Robotic Control" (arXiv 2606.26025) on top
of a copy of `hyunwoo`'s LeRobot checkout. **No training was run.** All 8 GPUs on this
machine are at 100% utilization / ~42-49GB memory used by another user's job, invisible in
our process namespace, for the entire duration of this work. Everything below was built and
verified on CPU only (or via static code inspection / config-only downloads).

## What's implemented

- `/workspace/yujin/icwm/lerobot/` -- a fresh git repo, copied from
  `/workspace/yujin/hyunwoo/lerobot` (excluding its 244GB `outputs/` and 14GB `.venv/`,
  neither of which is source code). `hyunwoo`'s original directory was never modified.
- `/workspace/yujin/icwm/lerobot/src/lerobot/policies/icwm/`:
  - `configuration_icwm.py` -- `ICWMConfig`: backbone (`Qwen/Qwen2.5-VL-3B-Instruct` by
    default), `n_context_clips=5`, `action_chunk_size=5`, FAST tokenizer settings, and the
    paper's optimizer/scheduler defaults (AdamW, `weight_decay=1e-4`, `lr=5e-5`, 50k-step
    warmup then cosine decay -- verified against a real download of the Qwen2.5-VL-3B
    `config.json`, see Verification below).
  - `modeling_icwm.py` -- `ICWMPolicy`: loads a **plain dense**
    `transformers.Qwen2_5_VLForConditionalGeneration` (not the MoE
    `Qwen2_5_VLMoEForAction` `wall_x` uses), resizes its token embedding to add FAST
    action-token ids, `forward()` (masked cross-entropy over target action tokens only),
    `predict_action_chunk()`/`select_action()` (autoregressive FAST-token decode via
    `model.generate()`), and a documented (not implemented) hook for the paper's Sec 6.4
    KV-cache-reuse-of-T optimization.
  - `processor_icwm.py` -- `ICWMSequenceBuilderStep`: builds the Fig. 2 interleaved sequence
    `[o^s_1, a_1, o^e_1] ... [o^s_N, a_N, o^e_N] [o_t, l] -> [a_t]` via the Qwen chat
    template, extends the Qwen vocab with `<|action_token_i|>` ids (following the same
    convention `lerobot.policies.wall_x` already uses for its own backbone), and computes
    the `labels` tensor so cross-entropy is supervised **only** on the target action-chunk
    token positions (paper Eq. 7), excluding the context-clip action tokens, images, and
    instruction text.
  - `__init__.py` and registration in
    `src/lerobot/policies/factory.py` (`get_policy_class`, `make_policy_config`, and the
    processor-pipeline dispatch), so `--policy.type=icwm` works with `lerobot_train.py`.
- `/workspace/yujin/icwm/lerobot/src/lerobot/datasets/icwm_context_sampler.py` --
  `ICWMInteractionSegmentPool` (pools candidate interaction segments across many
  trajectories/viewpoints, respecting episode boundaries via each dataset's
  `meta.episodes` table) and `ICWMContextSampledDataset` (wraps a task dataset, attaching N
  freshly-sampled clips per `__getitem__`). Column names (`dataset_from_index`,
  `dataset_to_index`, `length`) were checked directly against a real v3.0 LeRobot dataset on
  this machine (see Verification).
- `/workspace/yujin/icwm/scripts/`:
  - `render_libero_cross_view.sh` -- orchestration for rendering all 4 LIBERO suites at the
    8 in-domain azimuths (30/60/90/120/240/270/300/330). **Not run.** `libero_spatial` is
    already rendered (see Data status below); `libero_object`/`libero_goal`/`libero_10`
    are not, and the exact full-episode renderer to use for them was not identified within
    scope (see the script's header comment) -- only a frame-preview tool
    (`LIBERO-JY/xyg_scripts/render_theta_spatial.py`) and a non-CLI exploration script
    (`rotate_recolor_dataset.py`) were found.
  - `train_icwm_libero_table3.sh`, `train_mv_baseline_libero_table3.sh` -- launch configs
    for ICWM and the MV (`n_context_clips=0`) baseline. **Not run.**
  - `train_exp_baseline_libero_table3.sh` -- stub; EXP (MV + oracle camera azimuth as text)
    needs a small processor change plus per-sample azimuth metadata that the current
    rendered data doesn't carry (azimuth is only in the directory name). Documented as a
    follow-up, not implemented.

## Data status

- `/workspace/yujin/data/libero_spatial_rot/lerobot_v30/` already has LIBERO-Spatial
  rendered as separate LeRobot v3.0 datasets for `theta = 000, 030, 060, 090, 120, 240, 270,
  300, 330` -- i.e. exactly the paper's 8 in-domain training azimuths (plus 0). Confirmed by
  reading `meta/info.json` and a `meta/episodes/*.parquet` file directly.
- **The actual pipeline that produced it was found**: `/workspace/yujin/libero_rot/`
  (`render_shard.py` + `pack_lerobot.py` + `run_stage_a.sh`/`run_stage_b.sh`, also mirrored
  at `/workspace/yujin/FastWAM/libero_rot/`), driven by
  `/workspace/yujin/data/libero_spatial_rot/{stage_a,stage_b}.log` and per-task/per-theta
  logs under its `logs/` dir. This is a different, more complete tool than the
  `LIBERO-JY/xyg_scripts/` frame-preview script a previous pass found -- see
  `scripts/render_libero_cross_view.sh`'s header for the full trail (both `--suite` and
  `--raw-dir` were already CLI args, so no code changes were needed to generalize it to the
  other 3 suites).
- **LIBERO-Object, LIBERO-Goal, and LIBERO-10 (Long) rendering is now IN PROGRESS** using
  that exact same pipeline, launched via `scripts/render_libero_cross_view.sh`
  (`nohup`'d, PID tracked in `/workspace/yujin/data/_pipeline_logs/render_all.log`) at the
  paper's 8 in-domain azimuths (30/60/90/120/240/270/300/330 -- no theta000, since the task
  only asked for the 8). Raw hdf5 demonstrations were obtained via LIBERO's own official
  downloader (`libero.libero.utils.download_utils`, box.com links) for
  `libero_object`/`libero_goal` (10 tasks x 50 demos each, now at
  `/workspace/yujin/data/libero_raw/{libero_object,libero_goal}/`), and `libero_10`'s raw
  hdf5 was already present at `/workspace/yujin/data/libero_raw/libero_10/` (10 tasks x 50
  demos). A smoke test (1 demo, 2 thetas, `libero_object`) confirmed `render_shard.py` runs
  correctly against the newly-downloaded raw data before the full run was launched.
  Expected output, one LeRobot v3.0 dataset per (suite, theta), mirroring
  `libero_spatial_rot`'s schema exactly:
  - `/workspace/yujin/data/libero_object_rot/lerobot_v30/libero_object_theta{030,060,090,120,240,270,300,330}_no_noops_lerobot/`
  - `/workspace/yujin/data/libero_goal_rot/lerobot_v30/libero_goal_theta{...}_no_noops_lerobot/`
  - `/workspace/yujin/data/libero_10_rot/lerobot_v30/libero_10_theta{...}_no_noops_lerobot/`
  Per libero_rot's own README, rendering runs on CPU-only software (llvmpipe) MuJoCo
  offscreen rendering (no NVIDIA EGL/GL in this container) at ~4-5 renders/s/worker; with 30
  parallel workers per suite this took ~1.5-2h wall clock for libero_spatial's analogous
  9-theta run, so the 3 remaining suites (8 thetas each, run sequentially to avoid
  oversubscribing this machine's 31 CPUs) are expected to take on the order of several
  hours total. **Check `/workspace/yujin/data/<suite>_rot/stage_a.log` /
  `stage_b.log` and `/workspace/yujin/data/_pipeline_logs/render_all.log` for current
  progress/completion** -- this may still be running when this document is read.

## Verification performed (CPU only, no CUDA/GPU touched anywhere)

1. `python3 -m py_compile` on every new file -- all pass.
2. A CPU-only test venv (`/workspace/yujin/icwm/.testvenv312`, built from a Python 3.12
   interpreter found under `hyunwoo/lerobot/.venv/bin/python3.12` -- only that *interpreter
   binary* was reused read-only; the venv itself was created fresh, nothing in
   `hyunwoo/`'s files was modified) with `torch==2.14.0+cpu`, `transformers==5.16.1`,
   `scipy`, `draccus`, `gymnasium`, `datasets`, `einops`, `torchvision`, etc. installed
   (the base system Python only has `torch==1.9.0+cu111` / `transformers==4.8.1`, far too
   old for Qwen2.5-VL -- see Known gaps).
3. With that venv:
   - `from lerobot.policies.icwm import ICWMConfig, ICWMPolicy, make_icwm_pre_post_processors`
     and `lerobot.policies.factory.get_policy_class("icwm")` both succeed.
   - `ICWMConfig(...)` built against the **real** downloaded `Qwen/Qwen2.5-VL-3B-Instruct`
     `config.json` (config-only, no weights); `validate_features()`, `get_optimizer_preset()`
     (confirms `lr=5e-5`, `weight_decay=1e-4`), `get_scheduler_preset()` (confirms
     `num_warmup_steps=50000`), and `action_delta_indices` all checked.
   - `ICWMSequenceBuilderStep` constructed against the **real** Qwen tokenizer/processor and
     the **real** `lerobot/fast-action-tokenizer` (its actual API --
     `tokenizer(action_array)` -> list of per-sample token-id lists, and
     `tokenizer.decode(token_lists, time_horizon=, action_dim=)` -- was inspected directly
     and matches what `processor_icwm.py`/`modeling_icwm.py` assume; discovered its real
     vocab size is 2048, not the 1024 initially assumed -- fixed the config default).
   - A full `ICWMSequenceBuilderStep.complementary_data()` call with synthetic
     images/actions for 2 samples x 3 context clips produced a correctly shaped, correctly
     masked batch: supervised (`!= -100`) label positions are non-empty, all equal genuine
     `<|action_token_i|>` ids, and lie only in the trailing (target-action) span of each
     row's real (non-padded) tokens -- i.e. the context-clip action tokens earlier in the
     same sequence are correctly excluded from the loss.
   - `ICWMPolicy` was constructed end-to-end with a tiny randomly-initialized dense
     Qwen2.5-VL (2 layers, hidden size 16/32, real M-RoPE `mrope_section` config) via the
     `pretrained_path`/`_from_config` code path (no real weights downloaded), and both
     `forward()` (returns a finite masked cross-entropy loss) and `predict_action_chunk()`
     (autoregressive `generate()` decode -> correctly shaped `(1, 5, 7)` continuous action
     chunk) ran successfully on CPU.
   - This process caught and fixed one real bug: `Qwen2_5_VLConfig` no longer exposes a
     top-level `vocab_size` in `transformers>=5`; `modeling_icwm.py` now falls back to
     `config.text_config.vocab_size`.
4. `pip show transformers` in the base system environment: `4.8.1` -- **not** new enough for
   Qwen2.5-VL (needs modern `transformers`, empirically `5.16.1` worked; the paper-era
   `transformers>=4.49` line should also work but was not tested here). `torch` in the base
   environment is `1.9.0+cu111`, also far too old. Real training needs a proper environment
   upgrade; none was attempted on the base environment (only an isolated, throwaway test
   venv was created for verification, per the task's "don't resolve env issues" guidance).

## Before real training can start

1. Provision a training environment with modern `torch`/`transformers` (the base env here is
   too old) on a machine with actually-free GPUs.
2. Download `Qwen/Qwen2.5-VL-3B-Instruct` full weights and the `lerobot/fast-action-tokenizer`
   (only their configs/tokenizers were fetched here, config-only in accordance with the GPU
   safety constraint).
3. Render LIBERO-Object/Goal/Long at the 8 training azimuths and convert to LeRobot v3.0
   format (`scripts/render_libero_cross_view.sh` is scaffolded but not runnable as-is; see
   its header for the identified gap in available rendering tooling).
4. Wire `ICWMContextSampledDataset`/`ICWMInteractionSegmentPool` into an actual
   `lerobot_train.py` run (currently only unit-style-verified in isolation, not run against
   a real multi-GB dataset end-to-end).
5. Implement the EXP baseline properly (needs per-sample azimuth metadata; see
   `scripts/train_exp_baseline_libero_table3.sh`).

## Deviations from a literal reading of the paper (and why)

- **Backbone wiring pattern reused from `eo1`, not `wall_x`.** `wall_x` already has FAST
  tokenization + LeRobot dataset/processor plumbing, but its backbone
  (`Qwen2_5_VLMoEForAction`) is a Mixture-of-Experts variant, which the task explicitly
  requires avoiding. `eo1` uses the plain dense `Qwen2_5_VLForConditionalGeneration` (flow
  matching action head, not token-level), so ICWM's `modeling_icwm.py` follows `eo1`'s
  backbone-construction/`gradient_checkpointing`/`from_pretrained` vs. `_from_config`
  pattern, but replaces the flow-matching head entirely with FAST-token cross-entropy
  (structurally closer to `pi0_fast`, which also predicts discrete action tokens -- but
  `pi0_fast`'s backbone is PaliGemma, not Qwen).
- **Action-token vocabulary extension, not vocabulary reuse.** `pi0_fast` reuses the *top*
  of PaliGemma's existing (256k) vocabulary for FAST tokens (`vocab_size - 1 -
  fast_skip_tokens - token_id`), which only works because PaliGemma's tokenizer already has
  enough unused trailing ids. Qwen's tokenizer does not reserve such a block the same way,
  so ICWM instead adds genuinely new `<|action_token_i|>` special tokens and calls
  `resize_token_embeddings`, exactly mirroring `wall_x`'s own convention
  (`define_action_token_id`) for its MoE backbone. This satisfies the task's "extend/reuse
  Qwen tokenizer vocab with FAST action tokens, following how Pi-FAST does it" instruction
  in spirit (same token-space-sharing idea) while adapting the specific mechanism to Qwen.
- **KV-cache reuse of T (paper Sec 6.4) is not implemented**, only documented as a TODO in
  `ICWMPolicy.predict_action_chunk`'s docstring, per the task's explicit "don't
  over-engineer if time-limited" guidance. A full multi-timestep rollout today re-encodes T
  from scratch every `select_action` call; a real implementation needs a
  `precompute_context_kv(interaction_clips)` method plus cache threading through
  `select_action`.
- **Active Probing (physical/simulated random-reach data collection) is out of scope for
  `ICWMPolicy`** by design -- it is robot/env motor control, not a model computation. The
  policy's docstrings say the caller must supply already-collected `(o^s_i, a_i, o^e_i)`
  clips via the `ICWM_CONTEXT_*` observation keys; a separate env/robot-interface script
  (not written here) would own the actual random target-pose reaches.
