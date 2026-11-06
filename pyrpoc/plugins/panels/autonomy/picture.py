"""A survey's overview as a colour backdrop in dish µm: each channel stretched over its own range
and tinted its colour."""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QRectF

from pyrpoc.plugins.program_io.outputs import Placement
from pyrpoc.utils.widgets.colors import color_for_index

# Longest side drawn, in pixels: a whole-dish overview is far larger than any screen.
DISPLAY_PX = 2048
# Stretched between these percentiles, so a few bright pixels do not darken the rest.
STRETCH = (0.5, 99.8)


def backdrop(planes: np.ndarray, placement: Placement) -> tuple[np.ndarray, QRectF]:
    """``(C, H, W)`` planes as one ``(H, W, 3)`` image in 0..1, and the rectangle they cover."""
    step = max(1, math.ceil(max(planes.shape[1:]) / DISPLAY_PX))
    shown = planes[:, ::step, ::step]
    height, width = planes.shape[1:]
    rect = QRectF(
        placement.left_um,
        placement.top_um,
        width * placement.um_per_px,
        height * placement.um_per_px,
    )
    return tinted(shown), rect


def tinted(planes: np.ndarray) -> np.ndarray:
    height, width = planes.shape[1:]
    out = np.zeros((height, width, 3), dtype=np.float32)
    for index, plane in enumerate(planes):
        low, high = np.percentile(plane, STRETCH)
        scaled = np.clip((plane - low) / max(float(high - low), 1e-12), 0.0, 1.0)
        out += scaled[..., None] * (np.asarray(color_for_index(index), dtype=np.float32) / 255.0)
    return np.clip(out, 0.0, 1.0)
