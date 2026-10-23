#!/usr/bin/env python
"""Thin launcher wrapping `lerobot.scripts.lerobot_train`.

History: cuDNN used to be disabled here because conv3d (Qwen2.5-VL's vision patch embed) failed with
`CUDNN_STATUS_NOT_INITIALIZED` against the installed driver (535.274.02 / CUDA 12.2) with the
`torch==2.5.1+cu121` wheel's cuDNN. That failure no longer reproduces (re-tested 2026-10-06: conv3d
runs fine with cuDNN enabled), and leaving cuDNN off was not free: PyTorch's fallback
`slow_conv_dilated3d` cost ~0.9 s of every ~12 s training step (549 ms vs 4.4 ms per 4000 patches).
cuDNN is therefore left at its default (enabled). If the init error ever returns, set
ICWM_DISABLE_CUDNN=1 to get the old behaviour back.
"""

import torch

import os

if os.environ.get("ICWM_DISABLE_CUDNN"):
    torch.backends.cudnn.enabled = False

from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
