"""Measurements of one image channel: pixel statistics and histogram of a rectangle, and
the profile along a line. Positions are in pixel units, where pixel ``(i, j)`` covers
``[i, i + 1) x [j, j + 1)``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_HISTOGRAM_BINS = 128


@dataclass(frozen=True)
class RegionStats:
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class Histogram:
    counts: np.ndarray
    edges: np.ndarray


@dataclass(frozen=True)
class Profile:
    """Values sampled at even steps along a line, with each sample's distance from its start."""

    distance: np.ndarray
    values: np.ndarray
    length: float


def region_values(plane: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """The pixels of ``plane`` whose centres lie in the rectangle with corners ``(x0, y0)`` and
    ``(x1, y1)``, in either order."""
    height, width = plane.shape
    centres_x = np.arange(width) + 0.5
    centres_y = np.arange(height) + 0.5
    columns = np.flatnonzero((centres_x >= min(x0, x1)) & (centres_x <= max(x0, x1)))
    rows = np.flatnonzero((centres_y >= min(y0, y1)) & (centres_y <= max(y0, y1)))
    return plane[np.ix_(rows, columns)].ravel()


def region_stats(values: np.ndarray) -> RegionStats | None:
    """Summary of ``values``, or ``None`` when the region holds no pixels."""
    if values.size == 0:
        return None
    return RegionStats(
        count=int(values.size),
        mean=float(values.mean()),
        std=float(values.std()),
        minimum=float(values.min()),
        maximum=float(values.max()),
    )


def histogram(values: np.ndarray, bins: int | None) -> Histogram:
    """A histogram of ``values`` in ``bins`` bins, or as many as the data warrants for ``None``."""
    if bins is None:
        edges = np.histogram_bin_edges(values, bins="auto")
        bins = min(len(edges) - 1, MAX_HISTOGRAM_BINS)
    counts, edges = np.histogram(values, bins=bins)
    return Histogram(counts, edges)


def line_profile(
    plane: np.ndarray, start: tuple[float, float], end: tuple[float, float]
) -> Profile:
    """The values of ``plane`` along the line from ``start`` to ``end``, one sample per pixel of
    length, bilinearly interpolated. Samples outside the image are left out."""
    height, width = plane.shape
    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    samples = max(2, int(np.ceil(length)) + 1)
    fraction = np.linspace(0.0, 1.0, samples)
    # Shifted by half a pixel: a value belongs to its pixel's centre, not its corner.
    x = start[0] + fraction * (end[0] - start[0]) - 0.5
    y = start[1] + fraction * (end[1] - start[1]) - 0.5
    inside = (x >= -0.5) & (x <= width - 0.5) & (y >= -0.5) & (y <= height - 0.5)
    x = np.clip(x[inside], 0, width - 1)
    y = np.clip(y[inside], 0, height - 1)
    x0 = np.minimum(np.floor(x).astype(int), max(width - 2, 0))
    y0 = np.minimum(np.floor(y).astype(int), max(height - 2, 0))
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    values = (
        plane[y0, x0] * (1 - wx) * (1 - wy)
        + plane[y0, x1] * wx * (1 - wy)
        + plane[y1, x0] * (1 - wx) * wy
        + plane[y1, x1] * wx * wy
    )
    return Profile(fraction[inside] * length, values, length)
