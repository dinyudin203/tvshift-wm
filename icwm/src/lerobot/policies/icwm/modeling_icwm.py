#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""ICWM: In-Context World Modeling for Robotic Control (arXiv 2606.26025).

    a_t ~ pi_theta(a_t | Psi(T), o_t, l)

Psi is *not* a separate network. It shares parameters with the VLA backbone: Psi(T) is
just shorthand for "the hidden states the shared transformer produces after attending,
causally, over the N interaction-clip tokens". Concretely this policy:

  1. Uses a plain dense `transformers.Qwen2_5_VLForConditionalGeneration` backbone (the same
     wiring pattern as `lerobot.policies.eo1`, NOT the MoE backbone `lerobot.policies.wall_x`
     uses) -- see the module docstring in `configuration_icwm.py` for why.
  2. Represents actions as discrete FAST tokens co-resident in the Qwen vocabulary (see
     `processor_icwm.py`), so "conditioning on T" and "predicting a_t" are both just next-
     token prediction with the *same* backbone and the *same* vocabulary -- no extra modules,
     no extra loss term for Psi itself, exactly as the paper specifies.
  3. All of the interaction-context serialization happens upstream, in
     `processor_icwm.ICWMSequenceBuilderStep`; this module only needs to run the backbone
     forward pass (training) or the autoregressive FAST-token decode loop (inference).
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import _transformers_available, require_package

from ..pretrained import PreTrainedPolicy
from .configuration_icwm import ICWMConfig

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
else:
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None

logger = logging.getLogger(__name__)



def install_fast_patch_embed() -> bool:
    """Run Qwen2.5-VL's vision patch embedding as a Linear instead of a Conv3d.

    The patch embed is a Conv3d with kernel == stride == (2, 14, 14) applied to inputs that are
    already cut into exactly one kernel-sized patch each, so every output is a plain dot product of
    the flattened (C, T, 14, 14) patch with the flattened filter. On this machine cuDNN has to stay
    disabled (driver/cuDNN conv3d init failure), which leaves PyTorch's `slow_conv_dilated3d`
    (vol2col + gemm): ~0.9 s of every training step. The Linear form gives the same numbers up to
    float summation order, uses the same parameter (`proj.weight`, so checkpoints are unchanged)
    and runs as one tensor-core matmul.

    OFF by default: training keeps the original Conv3d (with cuDNN enabled, see run_icwm_train.py).
    Set ICWM_LINEAR_PATCH_EMBED=1 to opt in.
    """
    import os

    if not os.environ.get("ICWM_LINEAR_PATCH_EMBED"):
        return False
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen_vl

    cls = qwen_vl.Qwen2_5_VisionPatchEmbed
    if getattr(cls, "_icwm_linear_patch_embed", False):
        return True

    def forward(self, hidden_states):
        weight = self.proj.weight
        x = hidden_states.to(dtype=weight.dtype).reshape(-1, weight[0].numel())  # (patches, C*T*P*P)
        return torch.nn.functional.linear(x, weight.reshape(weight.shape[0], -1))

    cls.forward = forward
    cls._icwm_linear_patch_embed = True
    return True


install_fast_patch_embed()


class ICWMPolicy(PreTrainedPolicy):
    """ICWM policy: in-context conditioning on N (o^s_i, a_i, o^e_i) interaction clips."""

    config_class = ICWMConfig
    name = "icwm"

    def __init__(self, config: ICWMConfig, **kwargs):
        require_package("transformers", extra="icwm")
        super().__init__(config)
        config.validate_features()
        self.config = config

        dtype = None if config.dtype == "auto" else config.dtype
        if config.pretrained_path is None:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.backbone_name_or_path,
                dtype=dtype,
                attn_implementation=config.attn_implementation,
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration._from_config(
                config.vlm_backbone_config,
                dtype=config.vlm_backbone_config.dtype if dtype is None else dtype,
            )

        import os

        if os.environ.get("ICWM_VISION_FLASH_ATTN"):
            # The vision encoder attends inside small windows. With SDPA, HF splits the sequence and
            # runs one attention call per window (~12k tiny kernels per training step, launch-bound);
            # FlashAttention-2 does all windows of a layer in one variable-length call. Same exact
            # attention, different kernel; only the vision tower is switched, the LM keeps its kernel.
            # "1" -> the flash_attn package; any other value is passed through as the implementation
            # name, e.g. "kernels-community/flash-attn2" (prebuilt Hub kernel, no flash_attn build needed).
            impl = os.environ["ICWM_VISION_FLASH_ATTN"]
            if impl == "1":
                self.model.set_attn_implementation({"vision_config": "flash_attention_2"})
            else:  # the dict form rejects Hub kernel names; set it on the vision tower directly
                self.model.model.visual.set_attn_implementation(impl)
            logging.info("ICWM: vision encoder attention -> flash_attention_2 (LM unchanged: %s)",
                         self.model.config.text_config._attn_implementation)

        # Resize the token embedding matrix to account for the `<|action_token_i|>` ids
        # that `ICWMSequenceBuilderStep` adds to the Qwen tokenizer's vocabulary (see
        # processor_icwm.py). We recompute the target size directly (base Qwen vocab size +
        # action_token_vocab_size) so this works even if the processor step hasn't been
        # constructed yet (e.g. when only unit-testing the model in isolation).
        # `Qwen2_5_VLConfig` does not itself carry `vocab_size` in recent `transformers`
        # releases (only its nested `text_config` does); fall back to the top-level
        # attribute for older releases where it was mirrored there.
        base_vocab_size = getattr(self.model.config, "vocab_size", None)
        if base_vocab_size is None:
            base_vocab_size = self.model.config.text_config.vocab_size
        target_vocab_size = base_vocab_size + config.action_token_vocab_size
        if self.model.get_input_embeddings().num_embeddings < target_vocab_size:
            self.model.resize_token_embeddings(target_vocab_size)

        # FAST (DCT + BPE) action tokenizer -- used only at inference time in this class to
        # decode generated tokens back into continuous actions. Encoding of the *context*
        # clips and the *training* target happens upstream in `ICWMSequenceBuilderStep`
        # (kept there so preprocessing composes with LeRobot's standard processor pipeline).
        self._fast_tokenizer = AutoProcessor.from_pretrained(
            config.fast_tokenizer_name_or_path, trust_remote_code=True
        )
        # Action token i is "<|action_token_i|>", which the processor appends to the *tokenizer*, so its
        # id is len(tokenizer) + i (151665 + i for Qwen2.5-VL). The embedding matrix is larger than the
        # tokenizer (151936 rows, padded), so the offset must come from the tokenizer, not the config.
        from transformers import AutoTokenizer

        self._action_token_str_offset = len(AutoTokenizer.from_pretrained(config.backbone_name_or_path))
        self._im_end_id = AutoTokenizer.from_pretrained(config.backbone_name_or_path).convert_tokens_to_ids("<|im_end|>")

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # ICWM_GC_SKIP_LAYERS=N: keep the activations of the first N language-model decoder layers instead of
            # recomputing them (same math, trades spare VRAM for speed)
            n_skip = int(os.environ.get("ICWM_GC_SKIP_LAYERS", "0"))
            for layer in self.model.model.language_model.layers[:n_skip]:
                layer.gradient_checkpointing = False

        self.model.to(config.device)
        self.reset()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    # ------------------------------------------------------------------ training ----
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Standard next-token cross-entropy over the target action-chunk tokens only.

        `batch["labels"]` already has -100 everywhere except the target action-chunk token
        positions (see `ICWMSequenceBuilderStep.complementary_data`), so this is exactly
        `L = -log pi_theta(a_t | Psi(T), o_t, l)` (Eq. 7): HF's `Qwen2_5_VLForConditionalGeneration`
        computes the usual shifted-by-one causal-LM cross-entropy internally when `labels`
        is passed, and -100 positions (context clips, images, instruction) contribute zero
        loss, exactly matching the paper's masking.
        """
        if os.environ.get("ICWM_SPARSE_LOGITS") == "1":
            return self._forward_sparse_logits(batch)
        # Logits only from the position before the first supervised token to the end: the full
        # (B, L, vocab) fp32 logits of a long in-context sequence (10 clips: L ~ 2.6k, 3.2 GB at B=2)
        # do not fit next to the activations, and every other position has label -100 anyway.
        # Same loss as HF's causal-LM loss (shift by one, mean over the supervised tokens).
        all_labels = batch["labels"]
        seq_len = all_labels.shape[1]
        first_sup = int((all_labels != -100).float().argmax(1).min())
        keep = seq_len - first_sup + 1
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            logits_to_keep=keep,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1]  # positions seq_len - keep .. seq_len - 2
        labels = all_labels[:, seq_len - keep + 1 :]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        info = {"loss": loss.item()}
        with torch.no_grad():
            # Teacher-forced diagnostics over the supervised target-action tokens: a token is "right"
            # when the argmax of the previous position equals the label. A chunk only decodes when
            # *every* token is right (and the count matches), so exact_chunk_acc is the number to watch.
            pred = logits.argmax(-1)
            sup = labels != -100
            hit = (pred == labels) & sup
            n_sup = sup.sum().clamp(min=1)
            per_sample_ok = (hit.sum(1) == sup.sum(1)) & (sup.sum(1) > 0)
            info.update(
                {
                    "token_acc": (hit.sum() / n_sup).item(),
                    "exact_chunk_acc": per_sample_ok.float().mean().item(),
                    "target_tokens_per_chunk": (sup.sum() / sup.shape[0]).item(),
                    "seq_len": float(batch["attention_mask"].sum(1).float().mean().item()),
                }
            )
        return loss, info

    def _forward_sparse_logits(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Same loss and diagnostics as `forward`, but the LM head runs only at the positions that predict a
        supervised token (ICWM_SPARSE_LOGITS=1). The full [B, L, vocab] logits of a 5-clip sequence do not fit
        at batch 16; every other position has label -100 and contributes nothing to the loss anyway."""
        out = self.model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            use_cache=False,
        )
        labels = batch["labels"][:, 1:]
        sup = labels != -100
        h = out[0][:, :-1][sup]  # hidden state at position p predicts token p + 1
        logits = self.model.lm_head(h).float()
        tgt = labels[sup]
        loss = F.cross_entropy(logits, tgt)  # mean over supervised tokens = HF ForCausalLMLoss
        info = {"loss": loss.item()}
        with torch.no_grad():
            hit = torch.zeros_like(sup)
            hit[sup] = logits.argmax(-1) == tgt
            n_sup = sup.sum().clamp(min=1)
            per_sample_ok = (hit.sum(1) == sup.sum(1)) & (sup.sum(1) > 0)
            info.update(
                {
                    "token_acc": (hit.sum() / n_sup).item(),
                    "exact_chunk_acc": per_sample_ok.float().mean().item(),
                    "target_tokens_per_chunk": (sup.sum() / sup.shape[0]).item(),
                    "seq_len": float(batch["attention_mask"].sum(1).float().mean().item()),
                }
            )
        return loss, info

    # ----------------------------------------------------------------- inference ----
    def _decode_action_tokens_to_chunk(self, token_ids: list[int]) -> np.ndarray:
        """Inverse of `ICWMSequenceBuilderStep._encode_action_chunk`."""
        action_dim = self.config.output_features[ACTION].shape[0]
        return self._fast_tokenizer.decode(
            [token_ids], time_horizon=self.config.action_chunk_size, action_dim=action_dim
        )[0]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """In-context execution phase (paper Sec 4.2, part 2).

        The caller is responsible for the *active probing* phase: physically commanding N
        random target-pose reaches on the robot/env under the current system configuration
        psi and recording the resulting (o^s_i, a_i, o^e_i) triples. That is robot/env motor
        control, not a model computation, so it does not belong in this policy class -- it
        should live in a separate env/robot-interface script that then feeds the recorded
        clips into this batch via the `ICWM_CONTEXT_OBS_START/_OBS_END/_ACTIONS` observation
        keys (see `processor_icwm.py`), exactly like any other observation.

        Given that already-collected context T (Psi(T) is realized implicitly, for free, by
        attending over T in the forward pass below), this method just autoregressively
        decodes the FAST tokens for the current task query (o_t, l).

        KV-cache note (paper Sec 6.4): T is *static* under a fixed system configuration psi,
        so in a real multi-timestep rollout its hidden states/KV-cache can be computed once
        by running the backbone over T alone and reused for every subsequent timestep,
        instead of re-encoding T from scratch at every step. That optimization is NOT
        implemented here (it needs a second "prefill T only, then extend" code path plus
        cache management across `select_action` calls) -- functionally, a full forward pass
        over [T, o_t, l] every step (as done below) is equivalent, just slower. TODO: add a
        `precompute_context_kv(interaction_clips)` method that runs `self.model(..., T-only,
        use_cache=True)` once per new psi and threads the resulting `past_key_values` through
        `generate()`/manual decoding here.
        """
        self.eval()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        max_new_tokens = self.config.max_action_tokens_per_chunk if hasattr(
            self.config, "max_action_tokens_per_chunk"
        ) else 32
        gen_out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=self._im_end_id,
        )
        new_tokens = gen_out[:, input_ids.shape[1] :]

        action_dim = self.config.output_features[ACTION].shape[0]
        chunks = []
        for row in new_tokens.tolist():
            if self._im_end_id in row:  # the chunk ends at <|im_end|>
                row = row[: row.index(self._im_end_id)]
            action_ids = [t - self._action_token_str_offset for t in row if t >= self._action_token_str_offset]
            action_ids = [a for a in action_ids if 0 <= a < self.config.action_token_vocab_size]
            chunks.append(self._decode_action_tokens_to_chunk(action_ids))
        actions = torch.as_tensor(np.stack(chunks), dtype=torch.float32, device=input_ids.device)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()
