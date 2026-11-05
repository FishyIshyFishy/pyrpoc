"""A mosaic as one picture to show: the overview once stitched, or else the tiles placed by
their positions, shrunk to fit a display."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mosaic import Mosaic
from .stitching import composite, downsample


@dataclass(frozen=True)
class Picture:
    """A ``(C, H, W)`` image and the dish rectangle it covers, in µm."""

    planes: np.ndarray
    left_um: float
    top_um: float
    width_um: float
    height_um: float


def placed_tiles(mosaic: Mosaic, longest_px: int) -> Picture:
    """The tiles at their stage positions, blended, at most ``longest_px`` a side."""
    left, top, right, bottom = mosaic.extent_um()
    factor = max(1, math.ceil(max(right - left, bottom - top) / mosaic.um_per_px / longest_px))
    positions = np.array(
        [
            ((t.top_um - top) / mosaic.um_per_px, (t.left_um - left) / mosaic.um_per_px)
            for t in mosaic.tiles
        ]
    )
    planes, _ = composite([tile.image.array for tile in mosaic.tiles], positions, factor)
    scale = mosaic.um_per_px * factor
    return Picture(planes, left, top, planes.shape[2] * scale, planes.shape[1] * scale)


def overview_picture(mosaic: Mosaic, longest_px: int) -> Picture:
    """The stitched overview, at most ``longest_px`` a side."""
    overview = mosaic.overview
    if overview is None:
        raise ValueError("this mosaic has not been stitched")
    array = overview.image.array
    factor = max(1, math.ceil(max(array.shape[1:]) / longest_px))
    planes = downsample(array, factor)
    scale = overview.um_per_px * factor
    return Picture(
        planes, overview.left_um, overview.top_um, planes.shape[2] * scale, planes.shape[1] * scale
    )
