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
"""Pre/post-processing pipeline for the ICWM policy.

Builds, for each training/inference sample, the interleaved multimodal sequence described
in the paper's Fig. 2:

    [ o^s_1 image tokens, a_1 FAST-action tokens, o^e_1 image tokens ]   (clip 1)
    ...
    [ o^s_N image tokens, a_N FAST-action tokens, o^e_N image tokens ]   (clip N)
    [ o_t image tokens, instruction text tokens ]                        (task query)
    -> predict [ a_t FAST-action tokens ]                                (target, supervised)

Psi(T) is not computed by a separate module: it is simply "whatever hidden states the
shared Qwen2.5-VL backbone produces after attending, causally, over the N interaction
clips" -- i.e. this file only needs to *serialize* T + (o_t, l) into one token sequence; the
"conditioning" itself happens for free inside the backbone's ordinary forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_icwm import ICWMConfig

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
else:
    AutoProcessor = None
    Qwen2_5_VLProcessor = None

SYSTEM_MESSAGE = "You are a helpful physical assistant."

# Keys used to carry the N (o^s_i, a_i, o^e_i) interaction clips through the observation
# dict. These are populated upstream by
# `lerobot.datasets.icwm_context_sampler.ICWMContextSampledDataset` (see that module for the
# "pool of interaction segments collected across all training trajectories and viewpoints").
ICWM_CONTEXT_OBS_START = "observation.icwm_context.images_start"  # (N, C, H, W)
ICWM_CONTEXT_OBS_END = "observation.icwm_context.images_end"  # (N, C, H, W)
ICWM_CONTEXT_ACTIONS = "observation.icwm_context.actions"  # (N, action_chunk_size, action_dim)
# optional (B,) int: only the LAST n_valid of the N clips are real; the leading N - n_valid are
# placeholders (stacking needs a fixed N) and are left out of the sequence entirely
ICWM_CONTEXT_N_VALID = "observation.icwm_context.n_valid"
# optional second context segment, the query's own recent history (same clip format and n_valid
# convention). When present the two segments are tagged "Exploration:" / "History:" in the prompt.
ICWM_HISTORY_OBS_START = "observation.icwm_history.images_start"
ICWM_HISTORY_OBS_END = "observation.icwm_history.images_end"
ICWM_HISTORY_ACTIONS = "observation.icwm_history.actions"
ICWM_HISTORY_N_VALID = "observation.icwm_history.n_valid"


def make_action_token_str(token_id: int) -> str:
    return f"<|action_token_{token_id}|>"


@dataclass
@ProcessorStepRegistry.register(name="icwm_sequence_builder_processor")
class ICWMSequenceBuilderStep(ComplementaryDataProcessorStep):
    """Builds the full ICWM input sequence and, at training time, the action-only loss mask.

    This single step is responsible for:
      1. Encoding every continuous action chunk (the N context-clip actions, plus the target
         action chunk when present) into discrete FAST tokens.
      2. Assembling the Qwen2.5-VL chat-template `messages` for the interleaved
         [context clips] + [task query] sequence.
      3. Tokenizing everything with the (vocab-extended) Qwen processor.
      4. Computing `labels` so cross-entropy is only supervised on the target action-chunk
         token positions (Eq. 7: L = -log pi_theta(a_t | Psi(T), o_t, l)), never on the
         interaction-context tokens or the instruction/image tokens.
    """

    backbone_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    fast_tokenizer_name_or_path: str = "lerobot/fast-action-tokenizer"
    action_token_vocab_size: int = 2048
    max_action_tokens_per_chunk: int = 32
    image_min_pixels: int | None = 64 * 28 * 28
    image_max_pixels: int | None = 128 * 28 * 28
    use_fast_processor: bool = False
    input_features: dict[str, PolicyFeature] | dict[str, dict[str, Any]] = field(default_factory=dict)
    # context_source="mixed" prompt: "Exploration:" / "History:" only before non-empty segments, and "Task:"
    # before the query. False: both tags whenever a history segment is present, no "Task:" (n_history_clips).
    # Not in the processor json of mixed checkpoints saved before this field existed: the eval sets it from the config.
    mixed_markers: bool = False

    _processor: "Qwen2_5_VLProcessor | None" = field(default=None, init=False, repr=False)
    _fast_tokenizer: Any = field(default=None, init=False, repr=False)
    _action_token_ids: torch.Tensor | None = field(default=None, init=False, repr=False)
    _image_keys: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        require_package("transformers", extra="icwm")

        if self.input_features:
            first_val = next(iter(self.input_features.values()))
            if isinstance(first_val, dict):
                self.input_features = {
                    key: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
                    for key, v in self.input_features.items()
                }
        self._image_keys = [
            key for key, value in self.input_features.items() if value.type == FeatureType.VISUAL
        ]

        self._processor = Qwen2_5_VLProcessor.from_pretrained(
            self.backbone_name_or_path, use_fast=self.use_fast_processor
        )
        # FAST (Pertsch et al. 2501.09747) DCT+BPE action tokenizer, exactly the tokenizer
        # reused by lerobot's pi0_fast / wall_x "fast" prediction mode.
        self._fast_tokenizer = AutoProcessor.from_pretrained(
            self.fast_tokenizer_name_or_path, trust_remote_code=True
        )

        # Extend the shared Qwen vocabulary with dedicated FAST action-token ids, following
        # the same convention lerobot's wall_x policy uses for its own (MoE) backbone -- see
        # wall_x/modeling_wall_x.py::define_action_token_id. Reusing this convention keeps
        # ICWM's action tokens co-resident in the *same* vocabulary as text tokens.
        new_tokens = [make_action_token_str(i) for i in range(self.action_token_vocab_size)]
        self._processor.tokenizer.add_tokens(new_tokens, special_tokens=True)
        self._action_token_ids = torch.tensor(
            [self._processor.tokenizer.convert_tokens_to_ids(t) for t in new_tokens], dtype=torch.long
        )

    @property
    def processor(self) -> "Qwen2_5_VLProcessor":
        return self._processor

    @property
    def action_token_ids(self) -> torch.Tensor:
        return self._action_token_ids

    def _encode_action_chunk(self, action_chunk: torch.Tensor) -> str:
        """FAST-tokenize one continuous action chunk (T, action_dim) into a token string."""
        action_np = action_chunk.detach().cpu().numpy()[None]  # FAST tokenizer expects a batch dim
        token_ids = self._fast_tokenizer(action_np)[0]
        token_ids = list(token_ids[: self.max_action_tokens_per_chunk])
        return "".join(make_action_token_str(int(t)) for t in token_ids), len(token_ids)

    def complementary_data(self, complementary_data):
        tasks = complementary_data.get("task")
        if tasks is None:
            raise ValueError("Task is required for ICWMSequenceBuilderStep.")

        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required for ICWMSequenceBuilderStep.")
        action = self.transition.get(TransitionKey.ACTION)  # None at pure-inference time

        bsize = len(tasks)
        # one entry per context segment: (tag, starts, ends, actions, n_clips, n_valid per row)
        segments = []
        for tag, k_start, k_end, k_act, k_nv in (
            ("Exploration:", ICWM_CONTEXT_OBS_START, ICWM_CONTEXT_OBS_END, ICWM_CONTEXT_ACTIONS, ICWM_CONTEXT_N_VALID),
            ("History:", ICWM_HISTORY_OBS_START, ICWM_HISTORY_OBS_END, ICWM_HISTORY_ACTIONS, ICWM_HISTORY_N_VALID),
        ):
            acts = observation.get(k_act)
            if acts is None:
                continue
            n = acts.shape[1]
            nv = observation.get(k_nv)
            nv = [n] * bsize if nv is None else [int(v) for v in nv.reshape(-1).tolist()]
            segments.append([tag, observation[k_start], observation[k_end], acts, n, nv])
        tagged = any(seg[0] == "History:" for seg in segments)  # untagged: the original single-segment prompt

        # NOTE: by the time this step runs, `accelerate`'s `DataLoader` wrapper (via
        # `accelerator.prepare(dataloader)` in lerobot_train.py) may have already moved batch
        # tensors to the training device (e.g. CUDA) -- but the Qwen2.5-VL HF image
        # processor invoked below via `apply_chat_template` requires CPU/PIL/numpy inputs.
        # Force images (and the interaction-clip images) back to CPU here; the final
        # `DeviceProcessorStep` later in the pipeline moves the resulting tokenized tensors
        # back onto the training device.
        images_uint8 = {
            key: observation[key].detach().cpu().clamp(0, 1).mul(255.0).round().to(torch.uint8)
            for key in self._image_keys
            if key in observation
        }
        for seg in segments:
            seg[1:4] = [t.detach().cpu() for t in seg[1:4]]
        if action is not None:
            action = action.detach().cpu()

        messages = []
        target_lens = []
        for b in range(bsize):
            content: list[dict[str, Any]] = []
            # --- N interaction clips: [o^s_i tokens, a_i FAST tokens, o^e_i tokens] ---
            for tag, starts, ends, acts, n_clips, n_valid in segments:
                if tagged and (not self.mixed_markers or n_valid[b] > 0):
                    content.append({"type": "text", "text": tag})
                for i in range(n_clips - n_valid[b], n_clips):
                    start_img = starts[b, i].clamp(0, 1).mul(255.0).round().to(torch.uint8)
                    end_img = ends[b, i].clamp(0, 1).mul(255.0).round().to(torch.uint8)
                    action_text, _ = self._encode_action_chunk(acts[b, i])
                    content.append({"type": "image", "image": start_img})
                    content.append({"type": "text", "text": action_text})
                    content.append({"type": "image", "image": end_img})

            # --- Task query: current observation o_t + instruction l ---
            if tagged and self.mixed_markers:
                content.append({"type": "text", "text": "Task:"})
            for key in self._image_keys:
                if key in images_uint8:
                    content.append({"type": "image", "image": images_uint8[key][b]})
            cleaned_task = tasks[b].strip().replace("\n", " ")
            content.append({"type": "text", "text": cleaned_task})

            sample_messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {"role": "user", "content": content},
            ]

            if action is not None:
                target_text, target_len = self._encode_action_chunk(action[b])
                sample_messages.append(
                    {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
                )
                target_lens.append(target_len)
            messages.append(sample_messages)

        padding_side = "right" if action is not None else "left"
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            padding_side=padding_side,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
            add_generation_prompt=action is None,
            return_dict=True,
            return_tensors="pt",
        )

        complementary_data["input_ids"] = inputs["input_ids"]
        complementary_data["pixel_values"] = inputs["pixel_values"]
        complementary_data["image_grid_thw"] = inputs["image_grid_thw"]
        complementary_data["attention_mask"] = inputs["attention_mask"]

        if action is not None:
            # Loss mask: True only on the target action-chunk token positions (Eq. 7).
            # We restrict to positions carrying one of the reserved `<|action_token_i|>`
            # ids so that (a) the interaction-context action tokens embedded earlier in the
            # same sequence are excluded (they are conditioning input, not targets) and
            # (b) any chat-template control tokens around the assistant turn are excluded.
            # Because the assistant turn is the *last* message and action tokens are unique
            # ids, "is an action-token id AND lies at/after the assistant turn" isolates
            # exactly the target span; with right-padding the assistant span is the last
            # `target_len` non-pad tokens of the sequence for each row.
            # Loss mask (Eq. 7): every action token of the *assistant* turn plus the <|im_end|> that
            # closes it. The turn is located from its "<|im_start|>assistant\n" header rather than
            # by counting back from the end of the sequence: the chat template appends
            # "<|im_end|>\n" after the content, so counting back `target_len` tokens used to drop the
            # first two action tokens of every chunk and never taught the model to stop.
            # Context-clip action tokens sit in the user turn and stay excluded.
            input_ids = inputs["input_ids"]
            tok = self._processor.tokenizer
            im_start = tok.convert_tokens_to_ids("<|im_start|>")
            im_end = tok.convert_tokens_to_ids("<|im_end|>")
            header = len(tok("assistant\n", add_special_tokens=False)["input_ids"]) + 1  # <|im_start|> + "assistant\n"
            is_action_tok = torch.isin(input_ids, self._action_token_ids)
            labels = torch.full_like(input_ids, -100)
            for b in range(bsize):
                row = input_ids[b]
                starts = (row == im_start).nonzero().flatten()
                turn_start = int(starts[-1]) + header  # the assistant turn is the last one
                ends = (row[turn_start:] == im_end).nonzero().flatten()
                turn_end = turn_start + int(ends[0]) if len(ends) else int(inputs["attention_mask"][b].sum())
                mask = torch.zeros_like(row, dtype=torch.bool)
                mask[turn_start:turn_end] = is_action_tok[b, turn_start:turn_end]
                if turn_end < row.shape[0] and row[turn_end] == im_end:
                    mask[turn_end] = True  # learn to stop after the chunk
                n_act = int(is_action_tok[b, turn_start:turn_end].sum())
                if n_act != target_lens[b]:
                    raise RuntimeError(f"assistant turn has {n_act} action tokens, expected {target_lens[b]}")
                labels[b, mask] = row[mask]
            complementary_data["labels"] = labels

        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {
            "backbone_name_or_path": self.backbone_name_or_path,
            "fast_tokenizer_name_or_path": self.fast_tokenizer_name_or_path,
            "action_token_vocab_size": self.action_token_vocab_size,
            "max_action_tokens_per_chunk": self.max_action_tokens_per_chunk,
            "image_min_pixels": self.image_min_pixels,
            "image_max_pixels": self.image_max_pixels,
            "use_fast_processor": self.use_fast_processor,
            "mixed_markers": self.mixed_markers,
            "input_features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.input_features.items()
            },
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """This step only materializes tokenized tensors in complementary_data."""
        return features


def make_icwm_pre_post_processors(
    config: ICWMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build pre/post processor pipelines for ICWM."""

    context_norm_steps: list[ProcessorStep] = []
    ctx_action_keys = [k for k, n in ((ICWM_CONTEXT_ACTIONS, config.n_context_clips),
                                      (ICWM_HISTORY_ACTIONS, config.context_mixed_n_history
                                       if config.context_source == "mixed" else config.n_history_clips)) if n > 0]
    if config.normalize_context_actions and ctx_action_keys:
        # Context actions ride in the observation, where the normalizer skips ACTION-typed keys, so they are
        # registered as a STATE-typed feature carrying the ACTION norm mode and the target action's stats.
        context_norm_steps.append(
            NormalizerProcessorStep(
                features={k: PolicyFeature(type=FeatureType.STATE, shape=config.output_features[ACTION].shape)
                          for k in ctx_action_keys},
                norm_map={FeatureType.STATE: config.normalization_mapping["ACTION"]},
                stats=None if dataset_stats is None else {k: dataset_stats[ACTION] for k in ctx_action_keys},
                normalize_observation_keys=set(ctx_action_keys),
            )
        )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        *context_norm_steps,
        ICWMSequenceBuilderStep(
            backbone_name_or_path=config.backbone_name_or_path,
            fast_tokenizer_name_or_path=config.fast_tokenizer_name_or_path,
            action_token_vocab_size=config.action_token_vocab_size,
            max_action_tokens_per_chunk=config.max_action_tokens_per_chunk,
            image_min_pixels=config.image_min_pixels,
            image_max_pixels=config.image_max_pixels,
            use_fast_processor=config.use_fast_processor,
            input_features=config.input_features,
            mixed_markers=config.context_source == "mixed",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
