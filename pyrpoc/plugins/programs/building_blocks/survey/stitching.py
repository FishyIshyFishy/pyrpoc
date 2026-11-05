"""Stitching tiles whose positions are roughly known: registering overlapping neighbours, solving
for every tile's position at once, and blending them into one image."""

# Neighbouring tiles are registered pairwise by the normalised cross-correlation over their
# overlap, computed for every integer offset at once (Padfield's masked NCC): the cross term by
# zero-padded FFT, the overlap's sums from integral images. Unlike phase correlation there is no
# wrap-around to resolve and no edge artefact to outrank the true peak, which matters when tiles
# overlap only at their borders and hold smooth, blobby structure.
#
# A pair registers on whichever channel overlays it best, so one dim or empty channel cannot spoil
# it. The search is limited to near where the stage says the tiles are, so a repeated pattern
# cannot pull a tile far away; the positions are then solved by least squares over every pair,
# held weakly to the stage's, so a tile with no confident neighbour stays where the stage put it.

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Below this, a pair's best overlap is not trusted to be the same features.
MIN_SCORE = 0.3
# An overlap smaller than this share of a tile is too little to judge by.
MIN_OVERLAP_FRACTION = 0.01
MIN_OVERLAP_SIDE = 4
# How strongly a tile is held to its stage position, against a pair's score of up to 1.
STAGE_WEIGHT = 0.01

# ``(dy, dx)``: where the second tile's corner sits in the first tile's pixels.
Offset = tuple[float, float]


@dataclass(frozen=True)
class Registration:
    """Where tile ``second``'s corner sits in tile ``first``'s pixels, and how well."""

    first: int
    second: int
    offset: Offset
    score: float


def enough_overlap(rows: np.ndarray, cols: np.ndarray, height: int, width: int) -> np.ndarray:
    """Whether a ``rows`` x ``cols`` overlap of two tiles is enough to judge by."""
    return (
        (rows >= MIN_OVERLAP_SIDE)
        & (cols >= MIN_OVERLAP_SIDE)
        & (rows * cols >= MIN_OVERLAP_FRACTION * height * width)
    )


def downsample(plane: np.ndarray, factor: int) -> np.ndarray:
    """Block mean by ``factor``, dropping any remainder rows and columns."""
    if factor == 1:
        return plane
    height, width = plane.shape[-2] // factor, plane.shape[-1] // factor
    cropped = plane[..., : height * factor, : width * factor]
    blocks = cropped.reshape(*plane.shape[:-2], height, factor, width, factor)
    return blocks.mean(axis=(-3, -1))


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
    """The normalised cross-correlation of ``a`` and ``b`` over their overlap for every
    integer offset at once, indexed as ``padded_offsets`` says; offsets too small or flat to
    judge are -inf."""
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


def parabola_peak(left: float, centre: float, right: float) -> float:
    """The offset from the centre sample of the peak of the parabola through three samples."""
    curvature = left - 2.0 * centre + right
    if not math.isfinite(curvature) or curvature >= 0:
        return 0.0
    return 0.5 * (left - right) / curvature


def best_near(scores: np.ndarray, predicted: Offset, radius: float) -> tuple[Offset, float]:
    """The best-scoring offset within ``radius`` of ``predicted``, to a fraction of a pixel."""
    dy = padded_offsets(scores.shape[0] // 2)[:, None]
    dx = padded_offsets(scores.shape[1] // 2)[None, :]
    near = (dy - predicted[0]) ** 2 + (dx - predicted[1]) ** 2 <= radius**2
    masked = np.where(near, scores, -np.inf)
    y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
    score = float(masked[y, x])
    rows, cols = masked.shape
    sub_y = parabola_peak(masked[(y - 1) % rows, x], score, masked[(y + 1) % rows, x])
    sub_x = parabola_peak(masked[y, (x - 1) % cols], score, masked[y, (x + 1) % cols])
    return (float(dy[y, 0]) + sub_y, float(dx[0, x]) + sub_x), score


def register_near(
    first: int, second: int, frames: Sequence[np.ndarray], predicted: Offset, radius: float
) -> Registration | None:
    """Where ``(C, H, W)`` tile ``second`` sits relative to ``first``, within ``radius`` pixels
    of ``predicted``, on the channel that overlays them best; None when none is convincing."""
    best: Registration | None = None
    for channel in range(frames[first].shape[0]):
        scores = overlap_scores(frames[first][channel], frames[second][channel])
        offset, score = best_near(scores, predicted, radius)
        if score >= MIN_SCORE and (best is None or score > best.score):
            best = Registration(first, second, offset, score)
    return best


def solve_positions(stage_px: np.ndarray, pairs: Sequence[Registration]) -> np.ndarray:
    """Every tile's ``(top, left)`` in pixels that best agrees with the registered pairs,
    held weakly to ``stage_px``, the ``(N, 2)`` positions the stage gives."""
    count = len(stage_px)
    rows = len(pairs) + count
    system = np.zeros((rows, count))
    target = np.zeros((rows, 2))
    for row, pair in enumerate(pairs):
        weight = math.sqrt(pair.score)
        system[row, pair.second] = weight
        system[row, pair.first] = -weight
        target[row] = np.asarray(pair.offset) * weight
    anchor = math.sqrt(STAGE_WEIGHT)
    for index in range(count):
        system[len(pairs) + index, index] = anchor
        target[len(pairs) + index] = stage_px[index] * anchor
    solution, *_ = np.linalg.lstsq(system, target, rcond=None)
    return solution


def edge_weights(height: int, width: int) -> np.ndarray:
    """Weights highest at a tile's centre and falling linearly to each edge."""
    # So overlaps fade from one tile into the next instead of meeting at a seam.
    rows = np.minimum(np.arange(1, height + 1), np.arange(height, 0, -1)).astype(np.float32)
    cols = np.minimum(np.arange(1, width + 1), np.arange(width, 0, -1)).astype(np.float32)
    return np.outer(rows, cols)


def composite(
    frames: Sequence[np.ndarray], positions_px: np.ndarray, factor: int
) -> tuple[np.ndarray, tuple[float, float]]:
    """The ``(C, H, W)`` frames, at ``(top, left)`` ``positions_px``, shrunk by ``factor`` and
    blended into one image, and the ``(top, left)`` of its corner in the frames' pixels."""
    shrunk = [downsample(np.asarray(frame, dtype=np.float32), factor) for frame in frames]
    channels, height, width = shrunk[0].shape
    corner = positions_px.min(axis=0)
    placed = np.rint((positions_px - corner) / factor).astype(int)
    extent_y = int(placed[:, 0].max()) + height
    extent_x = int(placed[:, 1].max()) + width
    total = np.zeros((channels, extent_y, extent_x), dtype=np.float32)
    weight = np.zeros((extent_y, extent_x), dtype=np.float32)
    ramp = edge_weights(height, width)
    for frame, (y, x) in zip(shrunk, placed, strict=True):
        total[:, y : y + height, x : x + width] += ramp * frame
        weight[y : y + height, x : x + width] += ramp
    blended = np.divide(total, weight, out=np.zeros_like(total), where=weight > 0)
    return blended, (float(corner[0]), float(corner[1]))
