# Copyright 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Rolling FIFO buffer of completed interaction triplets (Appendix A.2).

    "At inference, the policy maintains a rolling FIFO buffer of completed triplets from
    the current episode: after each executed chunk, the newly formed triplet is appended
    and the oldest is evicted once the buffer is full. Triplets are drawn from the buffer
    with a fixed backward stride for deterministic input layout, and at the start of an
    episode the model simply conditions on whatever context is available."

A triplet only closes once its consequence has actually been observed, so the buffer holds a
*pending* observation between the moment a chunk is predicted and the moment the robot has
finished executing it.  Slots that the episode has not filled yet are padded and masked --
the same representation training uses for a frame that falls before the episode start, so
the model sees one consistent story.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Triplet:
    """``T_i = (tau_i, O_i, A_i, O_{i+C})``."""

    images: dict[str, Tensor]  # camera key -> [B, 3, H, W]
    state: Tensor  # [B, state_dim]
    action: Tensor  # [B, C, action_dim] -- the chunk that was actually executed
    consequence_images: dict[str, Tensor]
    consequence_state: Tensor
    step: int  # environment step of O_i


@dataclass
class _Pending:
    images: dict[str, Tensor]
    state: Tensor
    step: int
    action: Tensor | None = None


class TripletBuffer:
    """Holds up to ``capacity = K - 1`` completed triplets for one batched rollout."""

    def __init__(self, capacity: int, chunk_size: int):
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}")
        self.capacity = capacity
        self.chunk_size = chunk_size
        self._triplets: deque[Triplet] = deque(maxlen=max(capacity, 1))
        self._pending: _Pending | None = None
        self._step = 0

    def reset(self) -> None:
        """Called at every rollout boundary; context never crosses episodes."""
        self._triplets.clear()
        self._pending = None
        self._step = 0

    def __len__(self) -> int:
        return 0 if self.capacity == 0 else len(self._triplets)

    @property
    def step(self) -> int:
        return self._step

    # ------------------------------------------------------------------ recording

    def close_pending(self, images: dict[str, Tensor], state: Tensor) -> None:
        """Use the freshly reached observation as the consequence of the pending chunk."""
        pending = self._pending
        self._pending = None
        if pending is None or pending.action is None or self.capacity == 0:
            return
        self._triplets.append(
            Triplet(
                images={k: v.clone() for k, v in pending.images.items()},
                state=pending.state.clone(),
                action=pending.action.clone(),
                consequence_images={k: v.clone() for k, v in images.items()},
                consequence_state=state.clone(),
                step=pending.step,
            )
        )

    def open_pending(self, images: dict[str, Tensor], state: Tensor) -> None:
        """Record the observation the next chunk will be predicted from."""
        self._pending = _Pending(
            images={k: v.clone() for k, v in images.items()},
            state=state.clone(),
            step=self._step,
        )

    def commit_action(self, action: Tensor) -> None:
        """Attach the chunk that is about to be executed to the pending observation."""
        if self._pending is None:
            raise RuntimeError("commit_action() called without an open pending observation")
        self._pending.action = action.detach().clone()

    def advance(self, steps: int = 1) -> None:
        self._step += steps

    # --------------------------------------------------------------------- readout

    def context_inputs(
        self,
        current_images: dict[str, Tensor],
        current_state: Tensor,
        camera_keys: list[str],
        *,
        num_frames: int,
        action_dim: int,
    ) -> dict[str, Tensor]:
        """Stack the buffer plus the current observation into model inputs.

        Frames run oldest -> current along the frame axis.  Missing history occupies the
        *leading* slots and is masked out, so the current observation is always last.

        Returns keys ``obs_images`` / ``obs_image_masks`` / ``obs_states`` /
        ``consequence_*`` / ``context_actions`` / ``frame_masks`` / ``temporal_index``.
        """
        num_triplets = num_frames - 1
        available = list(self._triplets)[-num_triplets:] if num_triplets > 0 else []
        num_missing = num_triplets - len(available)

        reference = current_images[camera_keys[0]]
        bsize = reference.shape[0]
        device = reference.device

        def zeros_like_image(key: str) -> Tensor:
            return torch.zeros_like(current_images[key])

        obs_images: list[Tensor] = []
        cons_images: list[Tensor] = []
        for key in camera_keys:
            obs_stack, cons_stack = [], []
            for _ in range(num_missing):
                obs_stack.append(zeros_like_image(key))
                cons_stack.append(zeros_like_image(key))
            for triplet in available:
                obs_stack.append(triplet.images[key])
                cons_stack.append(triplet.consequence_images[key])
            obs_stack.append(current_images[key])
            obs_images.append(torch.stack(obs_stack, dim=1))
            if num_triplets > 0:
                cons_images.append(torch.stack(cons_stack, dim=1))

        state_stack = [torch.zeros_like(current_state) for _ in range(num_missing)]
        state_stack += [t.state for t in available]
        state_stack.append(current_state)
        obs_states = torch.stack(state_stack, dim=1)

        consequence_states = None
        context_actions = None
        if num_triplets > 0:
            cons_state_stack = [torch.zeros_like(current_state) for _ in range(num_missing)]
            cons_state_stack += [t.consequence_state for t in available]
            consequence_states = torch.stack(cons_state_stack, dim=1)

            zero_action = torch.zeros(
                bsize, self.chunk_size, action_dim, device=device, dtype=current_state.dtype
            )
            action_stack = [zero_action for _ in range(num_missing)]
            action_stack += [t.action.to(dtype=current_state.dtype) for t in available]
            context_actions = torch.stack(action_stack, dim=1)

        frame_masks = torch.ones(bsize, num_frames, dtype=torch.bool, device=device)
        if num_missing:
            frame_masks[:, :num_missing] = False

        # Backward offset in environment steps from the query frame; the current frame is 0.
        offsets = [0.0] * num_missing + [float(self._step - t.step) for t in available] + [0.0]
        temporal_index = torch.tensor(offsets, dtype=torch.float32, device=device)
        temporal_index = temporal_index[None].expand(bsize, num_frames).contiguous()

        ones = torch.ones(bsize, num_frames, dtype=torch.bool, device=device)
        out = {
            "obs_images": obs_images,
            "obs_image_masks": [ones.clone() for _ in camera_keys],
            "obs_states": obs_states,
            "frame_masks": frame_masks,
            "temporal_index": temporal_index,
        }
        if num_triplets > 0:
            cons_ones = torch.ones(bsize, num_triplets, dtype=torch.bool, device=device)
            out["consequence_images"] = cons_images
            out["consequence_image_masks"] = [cons_ones.clone() for _ in camera_keys]
            out["consequence_states"] = consequence_states
            out["context_actions"] = context_actions
        return out
