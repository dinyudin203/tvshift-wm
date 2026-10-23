"""A tiny stand-in for PaliGemmaWithExpertModel.

It keeps the real interface (``embed_image`` / ``embed_language_tokens`` / ``forward`` with a
4D additive mask over ``prefix ++ suffix``) and runs one genuine joint-attention layer, so
mask semantics are exercised end to end without instantiating a 3B backbone.
"""

from __future__ import annotations

import math
import types

import pytest
import torch
from torch import nn


class TinyPaliGemmaWithExpert(nn.Module):
    TOKENS_PER_IMAGE = 4

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision="float32",
        image_size=224,
        freeze_vision_encoder=False,
        train_expert_only=False,
    ):
        super().__init__()
        self.prefix_width = vlm_config.width
        self.suffix_width = action_expert_config.width
        dim = 64
        self.image_proj = nn.Linear(3 * 8 * 8, self.TOKENS_PER_IMAGE * self.prefix_width)
        self.token_embed = nn.Embedding(64, self.prefix_width)
        self.in_prefix = nn.Linear(self.prefix_width, dim)
        self.in_suffix = nn.Linear(self.suffix_width, dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_prefix = nn.Linear(dim, self.prefix_width)
        self.out_suffix = nn.Linear(dim, self.suffix_width)
        self.dim = dim

        # PI05Pytorch.forward probes this to decide whether to cast to bfloat16.
        probe = nn.Linear(1, 1)
        self.paligemma = types.SimpleNamespace(
            model=types.SimpleNamespace(
                language_model=types.SimpleNamespace(
                    layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(q_proj=probe))],
                    config=types.SimpleNamespace(_attn_implementation="eager"),
                )
            )
        )
        self.gemma_expert = types.SimpleNamespace(
            model=types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="eager"))
        )
        self._probe = probe

    def embed_image(self, image, **kwargs):
        flat = image.flatten(1)
        out = self.image_proj(flat)
        return out.view(image.shape[0], self.TOKENS_PER_IMAGE, self.prefix_width)

    def embed_language_tokens(self, tokens):
        return self.token_embed(tokens.clamp(min=0, max=63))

    def _attend(self, q, k, v, attention_mask):
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dim)
        if attention_mask is not None:
            scores = scores + attention_mask[:, 0].to(scores.dtype)
        return torch.softmax(scores, dim=-1) @ v

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        adarms_cond=None,
    ):
        from transformers import DynamicCache

        prefix, suffix = inputs_embeds

        if suffix is None:  # prefix prefill
            h = self.in_prefix(prefix)
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            out = self._attend(q, k, v, attention_mask)
            cache = DynamicCache(((k, v, None),)) if use_cache else None
            return [self.out_prefix(out), None], cache

        h_suffix = self.in_suffix(suffix)
        if adarms_cond is not None and adarms_cond[1] is not None:
            cond = adarms_cond[1]
            if cond.ndim == 2:  # pi05 emits one modulation per sample; ours is per token
                cond = cond[:, None]
            h_suffix = h_suffix + self.in_suffix(cond)

        if prefix is None:  # cached denoising step
            q, k, v = self.qkv(h_suffix).chunk(3, dim=-1)
            cached_k, cached_v, _ = next(iter(past_key_values))
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
            out = self._attend(q, k, v, attention_mask)
            return [None, self.out_suffix(out)], None

        h = torch.cat([self.in_prefix(prefix), h_suffix], dim=1)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        out = self._attend(q, k, v, attention_mask)
        p = prefix.shape[1]
        return [self.out_prefix(out[:, :p]), self.out_suffix(out[:, p:])], None


@pytest.fixture
def tiny_backbone(monkeypatch):
    import lerobot.policies.pi05.modeling_pi05 as pi05_mod

    monkeypatch.setattr(pi05_mod, "PaliGemmaWithExpertModel", TinyPaliGemmaWithExpert)
    return TinyPaliGemmaWithExpert
