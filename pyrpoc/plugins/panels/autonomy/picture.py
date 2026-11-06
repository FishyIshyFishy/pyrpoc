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
SAMPLE = 4


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
    scaled = np.empty(planes.shape, dtype=np.float32)
    for index, plane in enumerate(planes):
        # Every fourth pixel each way finds the same range as all of them, sixteen times faster.
        low, high = np.percentile(plane[::SAMPLE, ::SAMPLE], STRETCH)
        np.subtract(plane, low, out=scaled[index])
        scaled[index] *= 1.0 / max(float(high - low), 1e-12)
    np.clip(scaled, 0.0, 1.0, out=scaled)
    colours = np.asarray([color_for_index(i) for i in range(len(planes))], dtype=np.float32)
    # One matrix product mixes every channel's tint at once: (C, H, W) by (C, 3) to (H, W, 3).
    out = np.tensordot(scaled, colours / 255.0, axes=(0, 0))
    return np.clip(out, 0.0, 1.0, out=out)
