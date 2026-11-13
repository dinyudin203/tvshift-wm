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

import contextlib
import logging
import os
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.configs.types import FeatureType
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

        self.plucker_late_fusion = None
        if config.use_plucker:
            if getattr(config, "plucker_fusion", "early") == "late":
                from .plucker import install_plucker_late_fusion
                self.plucker_late_fusion = install_plucker_late_fusion(
                    self.model, getattr(config, "plucker_late_channels", 0),
                    mode=getattr(config, "plucker_late_mode", "concat"),
                    init_std=getattr(config, "plucker_late_init_std", 0.0))
            else:
                from .plucker import install_plucker_conditioning
                install_plucker_conditioning(self.model, getattr(config, "plucker_init", "zero"))
        if config.plucker_reasoning:
            self.plucker_head = torch.nn.Linear(self.model.config.text_config.hidden_size, 6,
                                          dtype=self.model.get_input_embeddings().weight.dtype)
        self._camera_rays = None
        if getattr(config, "use_camera_tokens", False):
            # Separate camera condition: MLP(Plucker ray) added to the <|camera_slot_i|> token embeddings only.
            # The last layer starts at zero, so the untrained policy equals the camera-free one.
            width = self.model.config.text_config.hidden_size
            dtype_ = self.model.get_input_embeddings().weight.dtype
            self.camera_token_mlp = torch.nn.Sequential(
                torch.nn.Linear(6, config.camera_token_hidden, dtype=dtype_), torch.nn.SiLU(),
                torch.nn.Linear(config.camera_token_hidden, config.camera_token_hidden, dtype=dtype_), torch.nn.SiLU(),
                torch.nn.Linear(config.camera_token_hidden, width, dtype=dtype_),
            )
            torch.nn.init.zeros_(self.camera_token_mlp[-1].weight)
            torch.nn.init.zeros_(self.camera_token_mlp[-1].bias)
            self.model.get_input_embeddings().register_forward_hook(self._add_camera_embeddings)

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

        self.action_expert = None
        if self._motion_ki():
            from .action_expert import FlowActionExpert

            text = self.model.config.text_config
            self.action_expert = FlowActionExpert(
                vlm_width=text.hidden_size, vlm_layers=text.num_hidden_layers,
                action_dim=config.output_features[ACTION].shape[0], chunk=config.action_chunk_size,
                width=config.action_expert_width, depth=config.action_expert_depth, heads=config.action_expert_heads,
            )
        self._last_motion_text: list[str] = []

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # ICWM_GC_SKIP_LAYERS=N: keep the activations of the first N language-model decoder layers instead of
            # recomputing them (same math, trades spare VRAM for speed)
            n_skip = int(os.environ.get("ICWM_GC_SKIP_LAYERS", "0"))
            for layer in self.model.model.language_model.layers[:n_skip]:
                layer.gradient_checkpointing = False
            n_vision_skip = int(os.environ.get("ICWM_VISION_GC_SKIP_BLOCKS", "0"))
            for block in self.model.model.visual.blocks[:n_vision_skip]:
                block.gradient_checkpointing = False

        if getattr(config, "freeze_vision_encoder", False):
            for p in self.model.model.visual.parameters():
                p.requires_grad_(False)
            print("vision encoder frozen:", sum(p.numel() for p in self.model.model.visual.parameters()), "params", flush=True)
        self.model.to(config.device)
        if self.plucker_late_fusion is not None:
            self.plucker_late_fusion.to(config.device)
        if config.plucker_reasoning:
            self.plucker_head.to(config.device)
        if getattr(config, "use_camera_tokens", False):
            self.camera_token_mlp.to(config.device)
        if self.action_expert is not None:
            self.action_expert.to(config.device)
        self.reset()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def _motion(self) -> bool:
        return bool(getattr(self.config, "motion_reasoning", False))

    def _motion_ki(self) -> bool:
        return self._motion() and getattr(self.config, "motion_reasoning_mode", "molmoact") == "ki"

    def _is_action_id(self, ids: Tensor) -> Tensor:
        lo = self._action_token_str_offset
        return (ids >= lo) & (ids < lo + self.config.action_token_vocab_size)

    def _camera_slot_range(self) -> tuple[int, int]:
        first = self._action_token_str_offset + self.config.action_token_vocab_size
        return first, first + self._n_camera_slots()

    def _n_camera_slots(self) -> int:
        views = sum(1 for ft in self.config.input_features.values() if ft.type is FeatureType.VISUAL)
        return views * self.config.camera_token_grid ** 2

    def _add_camera_embeddings(self, module, args, output):
        """Forward hook of the input embedding: + MLP(ray) at every camera-slot token, in slot order."""
        rays = self._camera_rays
        ids = args[0] if args else None
        if rays is None or ids is None:
            return output
        first, end = self._camera_slot_range()
        mask = (ids >= first) & (ids < end)
        if not mask.any():  # e.g. incremental decoding steps after the prompt
            return output
        rows, _ = mask.nonzero(as_tuple=True)
        encoded = self.camera_token_mlp(rays.to(device=output.device, dtype=output.dtype))  # (B, n_slots, width)
        delta = torch.zeros_like(output)
        delta[mask] = encoded[rows, ids[mask] - first]
        return output + delta

    @contextlib.contextmanager
    def _with_camera(self, batch):
        if not getattr(self.config, "use_camera_tokens", False):
            yield
            return
        rays = batch.get("camera_rays")
        if rays is None:
            raise ValueError("use_camera_tokens requires batch['camera_rays'] (camera calibration is missing)")
        first, end = self._camera_slot_range()
        ids = batch["input_ids"]
        n = ((ids >= first) & (ids < end)).sum(1)
        if (n != self._n_camera_slots()).any() or rays.shape[1] != self._n_camera_slots():
            raise RuntimeError(f"every row needs all {self._n_camera_slots()} camera slots, got {n.tolist()}")
        self._camera_rays = rays
        try:
            yield
        finally:
            self._camera_rays = None

    def get_optim_params(self) -> dict:
        expert_lr = getattr(self.config, "action_expert_lr", 0.0)
        if self.action_expert is not None and expert_lr:
            expert = {id(p) for p in self.action_expert.parameters()}
            rest = [p for p in self.parameters() if id(p) not in expert]
            return [{"params": rest}, {"params": list(self.action_expert.parameters()), "lr": expert_lr}]
        late_lr = getattr(self.config, "plucker_late_lr", 0.0)
        if self.plucker_late_fusion is None or not late_lr:
            return self.parameters()
        # The freshly initialised late-fusion module gets its own peak LR; the warmup/decay schedule scales
        # every group alike. Frozen parameters are left out exactly as optimizer.step would skip them.
        late = {id(p) for p in self.plucker_late_fusion.parameters()}
        rest = [p for p in self.parameters() if id(p) not in late]
        return [{"params": rest}, {"params": list(self.plucker_late_fusion.parameters()), "lr": late_lr}]

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
        with self._with_camera(batch):
            return self._forward_inner(batch)

    def _forward_inner(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self._motion():
            return self._forward_motion(batch)
        if os.environ.get("ICWM_SPARSE_LOGITS") == "1":
            return self._forward_sparse_logits(batch)
        if self.config.plucker_reasoning:
            raise ValueError("plucker_reasoning needs ICWM_SPARSE_LOGITS=1, which exposes the slot hidden states")
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            labels=batch["labels"],
            use_cache=False,
        )
        loss = outputs.loss
        info = {"loss": loss.item()}
        with torch.no_grad():
            # Teacher-forced diagnostics over the supervised target-action tokens: a token is "right"
            # when the argmax of the previous position equals the label. A chunk only decodes when
            # *every* token is right (and the count matches), so exact_chunk_acc is the number to watch.
            labels = batch["labels"][:, 1:]
            pred = outputs.logits[:, :-1].argmax(-1)
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
        if self.config.plucker_reasoning:
            reasoning_loss, reasoning_info = self._plucker_reasoning_loss(batch, out[0])
            info.update(action_loss=loss.item(), **reasoning_info)
            loss = loss + self.config.plucker_reasoning_loss_weight * reasoning_loss
            info["loss"] = loss.item()
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

    def _forward_motion(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Motion reasoning (sparse LM head, like _forward_sparse_logits).

        VLM loss = CE(action tokens + closing <|im_end|>) + w * CE(reasoning text), each a mean over its own
        tokens so the ~50 reasoning tokens do not drown the ~15 action tokens. "ki" adds the flow-matching
        loss of the action expert, which only sees detached VLM hidden states."""
        ki = self._motion_ki()
        out = self.model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            use_cache=False,
            output_hidden_states=ki,
        )
        labels = batch["labels"][:, 1:]
        sup = labels != -100
        h = out.last_hidden_state[:, :-1][sup]
        logits = self.model.lm_head(h).float()
        tgt = labels[sup]
        is_reason = ~self._is_action_id(tgt) & (tgt != self._im_end_id)
        if ki and not getattr(self.config, "ki_vlm_fast_loss", True):
            is_reason = torch.ones_like(is_reason)  # the turn is reasoning + <|im_end|>, no FAST tokens
        per_tok = F.cross_entropy(logits, tgt, reduction="none")
        reason_loss = per_tok[is_reason].mean() if is_reason.any() else per_tok.sum() * 0
        act_loss = per_tok[~is_reason].mean() if (~is_reason).any() else per_tok.sum() * 0
        loss = act_loss + self.config.motion_reasoning_loss_weight * reason_loss
        info = {"reasoning_loss": reason_loss.item(), "action_token_loss": act_loss.item()}
        if ki:
            key_mask = batch["attention_mask"].bool() & (
                torch.arange(batch["input_ids"].shape[1], device=batch["input_ids"].device)[None]
                < batch["reasoning_end"].to(batch["input_ids"].device)[:, None])
            action_dim = self.config.output_features[ACTION].shape[0]
            actions = batch[ACTION][..., :action_dim]
            flow = self.action_expert.loss(actions, out.hidden_states, key_mask, batch.get("action_is_pad"))
            loss = loss + self.config.action_expert_loss_weight * flow
            info["flow_loss"] = flow.item()
        info["loss"] = loss.item()
        with torch.no_grad():
            right = logits.argmax(-1) == tgt
            rows = sup.nonzero(as_tuple=True)[0]
            n = sup.shape[0]

            def per_row_all(sel):
                cnt = torch.zeros(n, device=tgt.device).index_add_(0, rows[sel], torch.ones_like(tgt[sel], dtype=torch.float))
                ok = torch.zeros(n, device=tgt.device).index_add_(0, rows[sel], right[sel].float())
                return ((ok == cnt) & (cnt > 0)).float().mean().item()

            info.update({
                "reasoning_token_acc": right[is_reason].float().mean().item() if is_reason.any() else 0.0,
                "reasoning_exact_acc": per_row_all(is_reason),
                "token_acc": right[~is_reason].float().mean().item() if (~is_reason).any() else 0.0,
                "exact_chunk_acc": per_row_all(~is_reason) if (~is_reason).any() else 0.0,
                "reasoning_tokens": is_reason.sum().item() / n,
                "seq_len": float(batch["attention_mask"].sum(1).float().mean().item()),
            })
        return loss, info

    def _plucker_reasoning_loss(self, batch: dict[str, Tensor], hidden: Tensor) -> tuple[Tensor, dict]:
        """Regress the query-view Plucker grids from the hidden states at the answer's slot tokens."""
        targets = batch["plucker_targets"].to(hidden.device).float()
        first = self._action_token_str_offset + self.config.action_token_vocab_size
        mask = (batch["input_ids"] >= first) & (batch["input_ids"] < first + targets.shape[1])
        if (mask.sum(1) != targets.shape[1]).any():
            raise RuntimeError("every row needs exactly one complete set of Plucker reasoning slots")
        pred = self.plucker_head(hidden[mask]).float().reshape(targets.shape)
        loss = F.mse_loss(pred, targets)
        with torch.no_grad():
            direction = (3, 6) if self.config.plucker_channel_order == "moment_direction" else (0, 3)
            moment = (0, 3) if direction == (3, 6) else (3, 6)
            cosine = F.cosine_similarity(pred[..., direction[0]:direction[1]],
                                         targets[..., direction[0]:direction[1]], dim=-1).mean()
            moment_error = (pred[..., moment[0]:moment[1]] - targets[..., moment[0]:moment[1]]).norm(dim=-1).mean()
        return loss, {"plucker_loss": loss.item(), "plucker_direction_cosine": cosine.item(),
                      "plucker_moment_l2": moment_error.item()}

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
        if self._motion():
            return self._predict_motion(batch, input_ids, attention_mask, max_new_tokens)
        with self._with_camera(batch):
            gen_out = self._generate(batch, input_ids, attention_mask, max_new_tokens)
        new_tokens = gen_out[:, input_ids.shape[1] :]
        return self._decode_generated(new_tokens, input_ids)

    def _predict_motion(self, batch, input_ids, attention_mask, max_action_tokens):
        """Reason, then act. molmoact: the FAST tokens generated after the reasoning are the action.
        ki: the reasoning tokens (up to the first action token / <|im_end|>) are appended to the prompt and
        the action expert integrates the flow from the VLM hidden states of prompt + reasoning."""
        ki = self._motion_ki()
        budget = self.config.motion_reasoning_max_tokens
        if not ki or getattr(self.config, "ki_vlm_fast_loss", True):
            budget += max_action_tokens
        gen_out = self._generate(batch, input_ids, attention_mask, budget)
        new_tokens = gen_out[:, input_ids.shape[1]:]
        reasoning = []
        for row in new_tokens.tolist():
            cut = len(row)
            for j, t in enumerate(row):
                if t == self._im_end_id or self._action_token_str_offset <= t < (
                        self._action_token_str_offset + self.config.action_token_vocab_size):
                    cut = j
                    break
            reasoning.append(row[:cut])
        from transformers import AutoTokenizer

        if not hasattr(self, "_text_tokenizer"):
            self._text_tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name_or_path)
        self._last_motion_text = [self._text_tokenizer.decode(r) for r in reasoning]
        if not ki:
            return self._decode_generated(new_tokens, input_ids)
        # prompt (left-padded by the processor) + reasoning, right-padded to a common length
        width = max(len(r) for r in reasoning)
        pad_id = self._text_tokenizer.pad_token_id or 0
        ext = torch.full((input_ids.shape[0], width), pad_id, dtype=input_ids.dtype, device=input_ids.device)
        ext_mask = torch.zeros_like(ext, dtype=attention_mask.dtype)
        for b, r in enumerate(reasoning):
            if r:
                ext[b, :len(r)] = torch.as_tensor(r, dtype=input_ids.dtype, device=input_ids.device)
                ext_mask[b, :len(r)] = 1
        ids = torch.cat([input_ids, ext], 1)
        mask = torch.cat([attention_mask, ext_mask], 1)
        out = self.model.model(input_ids=ids, attention_mask=mask, pixel_values=batch.get("pixel_values"),
                               image_grid_thw=batch.get("image_grid_thw"), use_cache=False, output_hidden_states=True)
        actions = self.action_expert.sample(out.hidden_states, mask.bool(), self.config.flow_num_steps)
        return actions.to(torch.float32)

    def _generate(self, batch, input_ids, attention_mask, max_new_tokens):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=self._im_end_id,
        )

    def _decode_generated(self, new_tokens, input_ids):

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
