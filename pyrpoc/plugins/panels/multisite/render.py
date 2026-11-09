"""Channels to colour: each channel scaled between its levels and added in its own colour."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PyQt6.QtGui import QImage, QPixmap

# The percentiles each channel's black and white are set at.
LOW_PERCENTILE, HIGH_PERCENTILE = 1.0, 99.7


def levels_of(planes: np.ndarray) -> np.ndarray:
    """Each channel's ``(black, white)``, as a ``(C, 2)`` array."""
    flat = planes.reshape(planes.shape[0], -1)
    low = np.percentile(flat, LOW_PERCENTILE, axis=1)
    high = np.percentile(flat, HIGH_PERCENTILE, axis=1)
    return np.stack([low, np.maximum(high, low + 1e-6)], axis=1)


def rgb_of(
    planes: np.ndarray, levels: np.ndarray, colours: Sequence[tuple[int, int, int]]
) -> np.ndarray:
    """A ``(C, H, W)`` image as ``(H, W, 3)`` bytes."""
    out = np.zeros((*planes.shape[1:], 3), dtype=np.float32)
    for plane, (low, high), colour in zip(planes, levels, colours, strict=True):
        scaled = np.clip((plane - low) / (high - low), 0.0, 1.0)
        out += scaled[..., None] * np.asarray(colour, dtype=np.float32)
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


def pixmap_of(rgb: np.ndarray) -> QPixmap:
    height, width, _ = rgb.shape
    image = QImage(rgb.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888)
    # The copy owns its pixels; the bytes it was made from go when this returns.
    return QPixmap.fromImage(image.copy())
