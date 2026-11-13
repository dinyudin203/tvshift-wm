"""Flow-matching action expert for knowledge-insulated (KI) ICWM.

pi0.5-KI / LAP style (LAP: `stop_action_to_vlm_grad`, which stop-gradients the VLM keys and values the
action expert attends to): the VLM is trained only by next-token losses (motion reasoning, optionally
FAST action tokens); the expert regresses the flow-matching velocity of the continuous action chunk while
reading the VLM's hidden states through `.detach()`, so no action-expert gradient ever reaches the VLM.

Each expert layer = bidirectional self-attention over the action tokens, cross-attention to the (detached,
layer-normed, projected) hidden states of one VLM layer (spread evenly over the VLM depth, last expert
layer <-> last VLM layer), MLP. The cross-attention keys are the prefix (context clips, query image,
instruction) and the motion-reasoning answer, never the FAST action tokens that may follow it.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sincos_time(t: Tensor, dim: int, max_period: float = 4.0, min_period: float = 4e-3) -> Tensor:
    """pi0's sine-cosine embedding of the flow time t in [0, 1]."""
    frac = torch.linspace(0.0, 1.0, dim // 2, device=t.device, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** frac
    x = t.to(torch.float64)[:, None] * (2 * math.pi / period)[None]
    return torch.cat([x.sin(), x.cos()], dim=-1).float()


class FlowActionExpert(nn.Module):
    def __init__(self, vlm_width: int, vlm_layers: int, action_dim: int, chunk: int,
                 width: int = 1024, depth: int = 8, heads: int = 16, mlp_ratio: int = 4):
        super().__init__()
        self.action_dim, self.chunk, self.width = action_dim, chunk, width
        # hidden_states[0] is the embedding output, hidden_states[i] the output of decoder layer i
        self.layer_ids = [max(1, round((i + 1) * vlm_layers / depth)) for i in range(depth)]
        self.ctx_norm = nn.ModuleList(nn.LayerNorm(vlm_width) for _ in range(depth))
        self.ctx_proj = nn.ModuleList(nn.Linear(vlm_width, width) for _ in range(depth))
        self.in_proj = nn.Linear(action_dim, width)
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.pos = nn.Parameter(torch.zeros(chunk, width))
        nn.init.normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(width, heads, mlp_ratio * width, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(depth)
        )
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, action_dim)

    def forward(self, x_t: Tensor, t: Tensor, hidden_states, key_mask: Tensor) -> Tensor:
        """x_t (B, chunk, D), t (B,), hidden_states: tuple of (B, L, vlm_width), key_mask (B, L) True = attend."""
        dtype = self.in_proj.weight.dtype
        h = self.in_proj(x_t.to(dtype)) + self.time_mlp(sincos_time(t, self.width).to(dtype))[:, None] + self.pos
        pad = ~key_mask.bool()
        for k, layer in enumerate(self.layers):
            ctx = hidden_states[self.layer_ids[k]].detach()  # knowledge insulation: no grad into the VLM
            mem = self.ctx_proj[k](self.ctx_norm[k](ctx.to(dtype)))
            h = layer(h, mem, memory_key_padding_mask=pad)
        return self.out(self.out_norm(h))

    def loss(self, actions: Tensor, hidden_states, key_mask: Tensor, is_pad: Tensor | None = None):
        """Conditional flow matching (pi0 convention): x_t = t*noise + (1-t)*a, target velocity noise - a."""
        b = actions.shape[0]
        actions = actions.float()
        noise = torch.randn_like(actions)
        t = torch.distributions.Beta(1.5, 1.0).sample((b,)).to(actions.device) * 0.999 + 0.001
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        v = self(x_t, t, hidden_states, key_mask).float()
        err = F.mse_loss(v, noise - actions, reduction="none").mean(-1)  # (B, chunk)
        if is_pad is not None:
            keep = (~is_pad.bool()).float()
            return (err * keep).sum() / keep.sum().clamp(min=1)
        return err.mean()

    @torch.no_grad()
    def sample(self, hidden_states, key_mask: Tensor, num_steps: int = 10) -> Tensor:
        b = key_mask.shape[0]
        x = torch.randn(b, self.chunk, self.action_dim, device=key_mask.device)
        dt = -1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((b,), 1.0 + i * dt, device=x.device)
            x = x + dt * self(x, t, hidden_states, key_mask).float()
        return x
