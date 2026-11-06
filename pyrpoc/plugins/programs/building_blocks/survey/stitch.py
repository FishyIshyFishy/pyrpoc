"""The survey's overview: neighbouring tiles registered near where the stage put them, every
position solved at once, and the tiles blended into one image."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pyrpoc.plugins.program_io.outputs import Placement

from .grid import Tile
from .stitching import (
    Registration,
    composite,
    register_near,
    solve_positions,
)

# The overview's longest side; a larger survey is stitched at a coarser pixel.
OVERVIEW_PX = 8192
# How precisely a pair registers, in pixels, when nothing moved between its two tiles.
REGISTRATION_PX = 0.25
# How far registration may move a tile from where the stage says, as a fraction of a tile,
# and at least this many pixels: enough for stage error and drift, too little to jump a cell.
SEARCH_FRACTION = 0.05
SEARCH_PX = 8.0


@dataclass(frozen=True)
class Overview:
    """The whole survey as one ``(C, H, W)`` image, and where it lies on the dish."""

    planes: np.ndarray
    placement: Placement


def neighbours(tiles: Sequence[Tile]) -> list[tuple[int, int]]:
    """Each pair of tiles one column or one row apart."""
    at = {(tile.row, tile.col): tile.index for tile in tiles}
    pairs = []
    for (row, col), index in at.items():
        for other in ((row, col + 1), (row + 1, col)):
            if other in at:
                pairs.append((index, at[other]))
    return pairs


def register_neighbours(
    frames: Sequence[np.ndarray], stage_px: np.ndarray, tiles: Sequence[Tile]
) -> list[Registration]:
    """Every neighbouring pair that registers convincingly near where the stage put it."""
    radius = max(SEARCH_PX, SEARCH_FRACTION * frames[0].shape[-1])
    found = []
    for first, second in neighbours(tiles):
        predicted = stage_px[second] - stage_px[first]
        pair = register_near(
            first, second, frames, (float(predicted[0]), float(predicted[1])), radius
        )
        if pair is not None:
            found.append(pair)
    return found


def axis_scale(pairs: list[Registration], stage_px: np.ndarray, axis: int) -> float:
    """How many pixels of content one pixel of stage move is along ``axis`` (0 is y, 1 is
    x), from the pairs that step along it; 1 with none."""
    # The stage is the ruler: what the calibration got slightly wrong about the scan's scale
    # shows up as content steps a steady fraction longer or shorter than the stage's.
    ratios = []
    for pair in pairs:
        step = stage_px[pair.second] - stage_px[pair.first]
        if abs(step[axis]) > abs(step[1 - axis]):
            ratios.append(pair.offset[axis] / step[axis])
    return float(np.median(ratios)) if ratios else 1.0


def in_stage_pixels(pairs: list[Registration], stage_px: np.ndarray) -> list[Registration]:
    """The pairs' offsets rescaled to the stage's pixels."""
    scale = (axis_scale(pairs, stage_px, 0), axis_scale(pairs, stage_px, 1))
    return [
        Registration(
            pair.first,
            pair.second,
            (pair.offset[0] / scale[0], pair.offset[1] / scale[1]),
            pair.score,
        )
        for pair in pairs
    ]


def trusted(
    pairs: list[Registration], times_s: Sequence[float], moved_px_per_s: float
) -> list[Registration]:
    """The pairs, each trusted less the further content could have moved between its two
    tiles."""
    # Its offset is uncertain by its own precision plus that motion, added in quadrature.
    weighted = []
    for pair in pairs:
        moved = moved_px_per_s * abs(times_s[pair.second] - times_s[pair.first])
        trust = REGISTRATION_PX**2 / (REGISTRATION_PX**2 + moved**2)
        weighted.append(Registration(pair.first, pair.second, pair.offset, pair.score * trust))
    return weighted


def stitch(
    frames: Sequence[np.ndarray],
    corners_um: Sequence[tuple[float, float]],
    times_s: Sequence[float],
    tiles: Sequence[Tile],
    um_per_px: float,
    um_per_min: float,
) -> tuple[Overview, list[tuple[float, float]]]:
    """The overview of the dish-oriented ``frames`` whose top-left pixels the stage put at
    ``corners_um`` (left, top) at ``times_s``, where content moves ``um_per_min``, and where
    stitching placed each corner instead."""
    left0 = min(left for left, _top in corners_um)
    top0 = min(top for _left, top in corners_um)
    stage_px = np.array(
        [((top - top0) / um_per_px, (left - left0) / um_per_px) for left, top in corners_um]
    )
    pairs = in_stage_pixels(register_neighbours(frames, stage_px, tiles), stage_px)
    pairs = trusted(pairs, times_s, um_per_min / 60.0 / um_per_px)
    positions = solve_positions(stage_px, pairs)
    extent = float(np.max(positions.max(axis=0) - positions.min(axis=0))) + frames[0].shape[-1]
    factor = max(1, math.ceil(extent / OVERVIEW_PX))
    planes, corner = composite(frames, positions, factor)
    placement = Placement(
        left0 + corner[1] * um_per_px, top0 + corner[0] * um_per_px, um_per_px * factor
    )
    overview = Overview(planes, placement)
    placed = [(left0 + float(x) * um_per_px, top0 + float(y) * um_per_px) for y, x in positions]
    return overview, placed
