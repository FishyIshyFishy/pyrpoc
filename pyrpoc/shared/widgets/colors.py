"""The channel colour palette shared by the overlay, mosaic and spectrum panels."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

PALETTE = [
    (255, 80, 80),
    (80, 220, 120),
    (70, 150, 255),
    (255, 200, 70),
    (190, 110, 255),
    (70, 230, 230),
    (255, 120, 210),
    (180, 180, 180),
]


def color_for_index(index: int) -> tuple[int, int, int]:
    """A channel's colour, stable across panels so it keeps its identity."""
    return PALETTE[index % len(PALETTE)]


def color_map_from_rgb(rgb: tuple[int, int, int]) -> pg.ColorMap:
    """Black to ``rgb``: one channel's LUT when channels are overlaid in colour."""
    r, g, b = rgb
    return pg.ColorMap(
        pos=np.array([0.0, 1.0], dtype=float),
        color=np.array([[0, 0, 0, 255], [r, g, b, 255]], dtype=np.ubyte),
    )
