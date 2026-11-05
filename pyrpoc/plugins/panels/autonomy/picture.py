"""A survey's mosaic as a colour backdrop: each channel stretched over its own range and tinted
its colour."""

from __future__ import annotations

import numpy as np

from pyrpoc.plugins.program_io.outputs import Mosaic
from pyrpoc.plugins.program_io.outputs.mosaic_view import Picture, overview_picture, placed_tiles
from pyrpoc.utils.widgets.colors import color_for_index

DISPLAY_PX = 2048
# Stretched between these percentiles, so a few bright pixels do not darken the rest.
STRETCH = (0.5, 99.8)


def backdrop(mosaic: Mosaic) -> Picture:
    """The overview, or the tiles where the stage put them while there is none."""
    if mosaic.overview is not None:
        return overview_picture(mosaic, DISPLAY_PX)
    return placed_tiles(mosaic, DISPLAY_PX)


def tinted(planes: np.ndarray) -> np.ndarray:
    """``(C, H, W)`` planes as one ``(H, W, 3)`` image in 0..1."""
    height, width = planes.shape[1:]
    out = np.zeros((height, width, 3), dtype=np.float32)
    for index, plane in enumerate(planes):
        low, high = np.percentile(plane, STRETCH)
        scaled = np.clip((plane - low) / max(float(high - low), 1e-12), 0.0, 1.0)
        out += scaled[..., None] * (np.asarray(color_for_index(index), dtype=np.float32) / 255.0)
    return np.clip(out, 0.0, 1.0)
