"""Stitching a mosaic's tiles from their content, without a pixel size.

Neighbouring tiles are registered pairwise by the normalised cross-correlation
over their overlap, computed for every integer offset at once (Padfield's
masked NCC): the cross term by zero-padded FFT, the overlap's sums from
integral images. Unlike phase correlation there is no wrap-around to resolve
and no edge artefact to outrank the true peak, which matters when tiles
overlap only at their borders and hold smooth, blobby structure.

A pair registers on whichever channel overlays it best, so one dim or empty
channel cannot spoil it, and the placement found applies to every channel.

Every stage step along an axis is the same move, so the grid is placed with one
step vector per axis: the median of that axis's confident pairs, refined at
full resolution. One vector per axis also absorbs a stage that runs against,
or slightly rotated from, the scan axes. Pairs register at a reduced working
resolution to keep a large mosaic quick; only the two steps are refined.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .layout import MosaicLayout

# Pairs register with tiles block-averaged down to at most this many pixels a side.
WORKING_PIXELS = 128
# Below this, a pair's best overlap is not trusted to be the same features.
MIN_SCORE = 0.3
# An overlap smaller than this share of a tile is too little to judge by.
MIN_OVERLAP_FRACTION = 0.01
MIN_OVERLAP_SIDE = 4
# Confident pairs per axis whose correlation the full-resolution refinement sums.
REFINE_PAIRS = 3

# ``(dy, dx)``: where the second tile's corner sits in the first tile's pixels.
Offset = tuple[int, int]
# Two neighbouring tiles' indices, the left or upper one first.
Pair = tuple[int, int]


@dataclass(frozen=True)
class Registration:
    offset: Offset
    score: float
    channel: int


@dataclass(frozen=True)
class AxisStep:
    """The step between neighbours along one grid axis, and how it was found:
    ``confident`` of ``registered`` pairs agreed, or none did and the step is
    the nominal one from the fallback overlap."""

    offset: Offset
    confident: int
    registered: int

    @property
    def nominal(self) -> bool:
        return self.confident == 0


@dataclass(frozen=True)
class GridSteps:
    col: AxisStep
    row: AxisStep


def overlap_slices(shape: tuple[int, int], offset: Offset) -> tuple[slice, slice, slice, slice]:
    """``(a_rows, a_cols, b_rows, b_cols)`` of the region two equal-shaped tiles
    share when the second's corner sits at ``offset`` in the first."""
    (height, width), (dy, dx) = shape, offset
    return (
        slice(max(0, dy), min(height, height + dy)),
        slice(max(0, dx), min(width, width + dx)),
        slice(max(0, -dy), min(height, height - dy)),
        slice(max(0, -dx), min(width, width - dx)),
    )


def enough_overlap(rows: np.ndarray, cols: np.ndarray, height: int, width: int) -> np.ndarray:
    """Whether a ``rows`` x ``cols`` overlap of two tiles is enough to judge by."""
    return (
        (rows >= MIN_OVERLAP_SIDE)
        & (cols >= MIN_OVERLAP_SIDE)
        & (rows * cols >= MIN_OVERLAP_FRACTION * height * width)
    )


def overlap_score(a: np.ndarray, b: np.ndarray, offset: Offset) -> float:
    """Normalised cross-correlation over the shared region, or -inf when it is
    too small or flat to judge."""
    height, width = a.shape
    dy, dx = offset
    if not enough_overlap(np.array(height - abs(dy)), np.array(width - abs(dx)), height, width):
        return -math.inf
    a_rows, a_cols, b_rows, b_cols = overlap_slices((height, width), offset)
    first = a[a_rows, a_cols] - a[a_rows, a_cols].mean()
    second = b[b_rows, b_cols] - b[b_rows, b_cols].mean()
    norm = float(np.sqrt((first * first).sum() * (second * second).sum()))
    if norm == 0.0:
        return -math.inf
    return float((first * second).sum()) / norm


def working_factor(shape: tuple[int, ...]) -> int:
    return max(1, math.ceil(max(shape) / WORKING_PIXELS))


def downsample(plane: np.ndarray, factor: int) -> np.ndarray:
    """Block mean by ``factor``, dropping any remainder rows and columns."""
    if factor == 1:
        return plane
    height, width = plane.shape[0] // factor, plane.shape[1] // factor
    blocks = plane[: height * factor, : width * factor].reshape(height, factor, width, factor)
    return blocks.mean(axis=(1, 3))


def padded_offsets(size: int) -> np.ndarray:
    """The offset each index of a correlation zero-padded to ``2 * size``
    stands for: 0 up to ``size - 1``, then the negatives."""
    index = np.arange(2 * size)
    return np.where(index < size, index, index - 2 * size)


def region_sums(
    plane: np.ndarray, rows: tuple[np.ndarray, np.ndarray], cols: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """The sum of ``plane[r0:r1, c0:c1]`` for every pair of row and column
    bounds: each offset's rows summed into a strip, then its columns."""
    (r0, r1), (c0, c1) = rows, cols
    down = np.zeros((plane.shape[0] + 1, plane.shape[1]))
    down[1:] = plane.cumsum(axis=0)
    strips = down[r1] - down[r0]
    across = np.zeros((strips.shape[0], plane.shape[1] + 1))
    across[:, 1:] = strips.cumsum(axis=1)
    return across[:, c1] - across[:, c0]


def overlap_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``overlap_score`` for every integer offset at once, indexed as
    ``padded_offsets`` says; offsets too small or flat to judge are -inf."""
    height, width = a.shape
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    dy, dx = padded_offsets(height), padded_offsets(width)
    a_rows = (np.clip(dy, 0, height), np.clip(height + dy, 0, height))
    a_cols = (np.clip(dx, 0, width), np.clip(width + dx, 0, width))
    b_rows = (np.clip(-dy, 0, height), np.clip(height - dy, 0, height))
    b_cols = (np.clip(-dx, 0, width), np.clip(width - dx, 0, width))

    rows, cols = a_rows[1] - a_rows[0], a_cols[1] - a_cols[0]
    count = np.outer(rows, cols).astype(np.float64)
    enough = enough_overlap(rows[:, None], cols[None, :], height, width)
    count[~enough] = 1.0
    sum_a, sum_b = region_sums(a, a_rows, a_cols), region_sums(b, b_rows, b_cols)
    variance_a = region_sums(a * a, a_rows, a_cols) - sum_a**2 / count
    variance_b = region_sums(b * b, b_rows, b_cols) - sum_b**2 / count

    shape = (2 * height, 2 * width)
    spectrum = np.fft.rfft2(a, s=shape) * np.conj(np.fft.rfft2(b, s=shape))
    covariance = np.fft.irfft2(spectrum, s=shape) - sum_a * sum_b / count

    # Relative to the overlap's size, so rounding in the sums never reads as texture.
    usable = enough & (variance_a > 1e-9 * count) & (variance_b > 1e-9 * count)
    scores = np.full(shape, -np.inf)
    scores[usable] = covariance[usable] / np.sqrt(variance_a[usable] * variance_b[usable])
    return scores


def register_planes(a: np.ndarray, b: np.ndarray) -> tuple[Offset, float]:
    """The offset that best overlays plane ``b`` on plane ``a``, in
    full-resolution pixels, and its score. Every offset is scored at the
    working resolution and the best kept."""
    factor = working_factor(a.shape)
    scores = overlap_scores(downsample(a, factor), downsample(b, factor))
    y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    dy = int(padded_offsets(scores.shape[0] // 2)[y])
    dx = int(padded_offsets(scores.shape[1] // 2)[x])
    return (dy * factor, dx * factor), float(scores[y, x])


def register_pair(first: np.ndarray, second: np.ndarray) -> Registration | None:
    """Where ``(C, H, W)`` tile ``second`` sits relative to ``first``, on the
    channel that overlays them best, or None when no channel is convincing."""
    best: Registration | None = None
    for channel in range(first.shape[0]):
        offset, score = register_planes(
            np.asarray(first[channel], dtype=np.float32),
            np.asarray(second[channel], dtype=np.float32),
        )
        if score >= MIN_SCORE and (best is None or score > best.score):
            best = Registration(offset, score, channel)
    return best


def neighbour_pairs(layout: MosaicLayout) -> tuple[list[Pair], list[Pair]]:
    """``(horizontal, vertical)``: neighbours one column apart, and one row apart."""
    at = {(tile.row, tile.col): tile.index for tile in layout.tiles}
    horizontal, vertical = [], []
    for (row, col), index in at.items():
        if (row, col + 1) in at:
            horizontal.append((index, at[row, col + 1]))
        if (row + 1, col) in at:
            vertical.append((index, at[row + 1, col]))
    return horizontal, vertical


def refine_step(planes: list[tuple[np.ndarray, np.ndarray]], start: Offset, radius: int) -> Offset:
    """The offset within ``radius`` of ``start`` that best overlays every pair."""
    best, best_score = start, -math.inf
    for dy in range(start[0] - radius, start[0] + radius + 1):
        for dx in range(start[1] - radius, start[1] + radius + 1):
            score = sum(overlap_score(a, b, (dy, dx)) for a, b in planes)
            if score > best_score:
                best, best_score = (dy, dx), score
    return best


class Stitcher:
    """One dataset's tiles. Pair registrations are kept, so a tile arriving
    mid-run registers only against its acquired neighbours."""

    def __init__(self, layout: MosaicLayout):
        self.layout = layout
        self.horizontal, self.vertical = neighbour_pairs(layout)
        self._registered: dict[Pair, Registration | None] = {}
        # Keyed by the pairs refined against and the coarse step, which a new
        # tile mid-run rarely changes, so re-stitching skips the search.
        self._refined: dict[tuple[tuple[Pair, ...], Offset], Offset] = {}

    def steps(self, frames: Sequence[np.ndarray], fallback_overlap: float) -> GridSteps:
        height, width = frames[0].shape[1:]
        return GridSteps(
            col=self.axis_step(frames, self.horizontal, (0, round(width * (1 - fallback_overlap)))),
            row=self.axis_step(frames, self.vertical, (round(height * (1 - fallback_overlap)), 0)),
        )

    def axis_step(
        self, frames: Sequence[np.ndarray], pairs: list[Pair], nominal: Offset
    ) -> AxisStep:
        acquired = [pair for pair in pairs if max(pair) < len(frames)]
        for first, second in acquired:
            if (first, second) not in self._registered:
                self._registered[first, second] = register_pair(frames[first], frames[second])
        confident = [
            (pair, found) for pair in acquired if (found := self._registered[pair]) is not None
        ]
        if not confident:
            return AxisStep(nominal, 0, len(acquired))

        dy, dx = np.median([found.offset for _pair, found in confident], axis=0)
        strongest = sorted(confident, key=lambda item: item[1].score, reverse=True)
        chosen = tuple(pair for pair, _found in strongest[:REFINE_PAIRS])
        key = (chosen, (round(dy), round(dx)))
        if key not in self._refined:
            planes = [
                (
                    np.asarray(frames[a][found.channel], dtype=np.float32),
                    np.asarray(frames[b][found.channel], dtype=np.float32),
                )
                for (a, b), found in strongest[:REFINE_PAIRS]
            ]
            radius = working_factor(frames[0].shape[1:])
            self._refined[key] = refine_step(planes, key[1], radius)
        return AxisStep(self._refined[key], len(confident), len(acquired))


def tile_positions(layout: MosaicLayout, steps: GridSteps, count: int) -> dict[int, Offset]:
    """The top-left pixel of each of the first ``count`` tiles, with the
    mosaic's own top-left at (0, 0)."""
    raw = {
        tile.index: (
            tile.col * steps.col.offset[0] + tile.row * steps.row.offset[0],
            tile.col * steps.col.offset[1] + tile.row * steps.row.offset[1],
        )
        for tile in layout.tiles
        if tile.index < count
    }
    top = min(y for y, _x in raw.values())
    left = min(x for _y, x in raw.values())
    return {index: (y - top, x - left) for index, (y, x) in raw.items()}


def edge_weights(height: int, width: int) -> np.ndarray:
    """Highest at the centre and falling linearly to each edge, so overlaps
    fade from one tile into the next instead of meeting at a seam."""
    rows = np.minimum(np.arange(1, height + 1), np.arange(height, 0, -1)).astype(np.float32)
    cols = np.minimum(np.arange(1, width + 1), np.arange(width, 0, -1)).astype(np.float32)
    return np.outer(rows, cols)


def composite(frames: Sequence[np.ndarray], positions: dict[int, Offset]) -> np.ndarray:
    """The placed ``(C, H, W)`` tiles blended into one ``(C, H', W')`` image.
    Pixels no tile covers are 0."""
    channels, height, width = frames[0].shape
    extent_y = max(y for y, _x in positions.values()) + height
    extent_x = max(x for _y, x in positions.values()) + width
    total = np.zeros((channels, extent_y, extent_x), dtype=np.float32)
    weight = np.zeros((extent_y, extent_x), dtype=np.float32)
    ramp = edge_weights(height, width)
    for index, (y, x) in positions.items():
        total[:, y : y + height, x : x + width] += ramp * frames[index]
        weight[y : y + height, x : x + width] += ramp
    return np.divide(total, weight, out=np.zeros_like(total), where=weight > 0)
