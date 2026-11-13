"""Pixel-axis motion reasoning targets.

Question the policy answers before its action: "in THIS camera, which delta-EEF moves the gripper
X cm along the image's +x (right), +y (down) and +z (away from the camera)?"

The pixel axes are the camera's OpenCV axes: +x = image right (+u), +y = image down (+v),
+z = optical axis into the scene. Moving along camera +x / +y at constant depth moves the EEF's
projection purely along +u / +v. The answer is a translation in the robot-base frame (the frame of
LIBERO's delta-EEF actions); rotation and gripper deltas are zero and left out.

It depends only on the camera rotation, so it is the same for every frame of a viewpoint: the model
has to read the viewpoint off the interaction context (and the current image) to get it right.

Units (``motion_reasoning_units``):
  * "context_step": raw action units per step, spread over one context clip (``horizon`` steps).
  * "chunk_step":   raw action units per step, spread over one predicted action chunk
                    (``action_chunk_size`` steps). With ICWM's context horizon == action_chunk_size
                    this is numerically identical to "context_step".
  * "total":        raw action units summed over the whole move (sum of per-step actions).
  * "cm":           the metric translation itself, in cm (LAP-style language numbers).
Raw action -> metres uses a per-axis gain (m per action unit per step) measured on the training data
(least squares of the EEF displacement over 5 steps on the summed actions).
"""
from __future__ import annotations

import re

import torch

AXES = ("x", "y", "z")
UNITS = ("context_step", "chunk_step", "total", "cm")
# OpenGL camera axes (x right, y up, looking down -z) -> pixel/OpenCV axes (x right, y down, z forward)
_GL_TO_PIXEL = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))


def pixel_axes_in_base(camera_to_world: torch.Tensor, base_to_world: torch.Tensor | None = None) -> torch.Tensor:
    """(3, 3) matrix whose ROWS are the image +x, +y, +z directions expressed in the base frame."""
    R = camera_to_world[:3, :3].to(torch.float64) @ _GL_TO_PIXEL
    if base_to_world is not None:
        R = base_to_world[:3, :3].to(torch.float64).T @ R
    return R.T


def motion_targets(camera_to_world, base_to_world, cm: float, units: str, gain, horizon: int) -> torch.Tensor:
    """(3 axes, 3 components) delta-EEF for an X-cm move along each pixel axis, in `units`."""
    d = pixel_axes_in_base(camera_to_world, base_to_world) * (cm / 100.0)  # metres, rows = pixel axes
    if units == "cm":
        return (d * 100.0).float()
    a = d / torch.as_tensor(gain, dtype=torch.float64)[None]  # total raw action units
    if units == "total":
        return a.float()
    if units in ("context_step", "chunk_step"):
        return (a / horizon).float()
    raise ValueError(f"unknown motion_reasoning_units {units!r}; expected one of {UNITS}")


def format_motion(values: torch.Tensor, decimals: int) -> str:
    """'x:+0.92,+0.01,-0.10;y:...;z:...;' -- signs always written so every number tokenizes the same way."""
    rows = []
    for name, row in zip(AXES, values.tolist(), strict=True):
        rows.append(f"{name}:" + ",".join(f"{v:+.{decimals}f}" for v in row))
    return ";".join(rows) + ";"


_NUM = r"([+-]\d+(?:\.\d+)?)"
_ROW = re.compile(rf"([xyz]):{_NUM},{_NUM},{_NUM}")


def parse_motion(text: str) -> torch.Tensor | None:
    """Inverse of format_motion; None when an axis is missing or malformed."""
    found = {m.group(1): [float(m.group(i)) for i in (2, 3, 4)] for m in _ROW.finditer(text)}
    if set(found) != set(AXES):
        return None
    return torch.tensor([found[a] for a in AXES])
