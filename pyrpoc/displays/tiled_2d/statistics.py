"""Pure-numpy image-analysis math for the tiled display.

Everything here operates on plain arrays and scalar geometry so it can be unit
tested without Qt. Coordinates follow the display convention: data is row-major
``[H, W]`` with ``x`` the column and ``y`` the row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegionStats:
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    total: float


def empty_stats() -> RegionStats:
    return RegionStats(count=0, mean=0.0, std=0.0, minimum=0.0, maximum=0.0, total=0.0)


def rectangle_mask(height: int, width: int, x: float, y: float, w: float, h: float) -> np.ndarray:
    """Boolean mask of the axis-aligned rectangle with corner (x, y) and size (w, h)."""
    mask = np.zeros((height, width), dtype=bool)
    x0 = int(np.clip(np.floor(min(x, x + w)), 0, width))
    x1 = int(np.clip(np.ceil(max(x, x + w)), 0, width))
    y0 = int(np.clip(np.floor(min(y, y + h)), 0, height))
    y1 = int(np.clip(np.ceil(max(y, y + h)), 0, height))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def ellipse_mask(height: int, width: int, x: float, y: float, w: float, h: float) -> np.ndarray:
    """Boolean mask of the axis-aligned ellipse inscribed in the (x, y, w, h) box."""
    cx = x + w / 2.0
    cy = y + h / 2.0
    rx = abs(w) / 2.0
    ry = abs(h) / 2.0
    if rx <= 0.0 or ry <= 0.0:
        return np.zeros((height, width), dtype=bool)
    cols = np.arange(width, dtype=np.float64)[np.newaxis, :] + 0.5
    rows = np.arange(height, dtype=np.float64)[:, np.newaxis] + 0.5
    nx = (cols - cx) / rx
    ny = (rows - cy) / ry
    return (nx * nx + ny * ny) <= 1.0


def region_stats(channel: np.ndarray, mask: np.ndarray) -> RegionStats:
    """Summary statistics for the pixels of ``channel`` selected by ``mask``."""
    values = channel[mask]
    if values.size == 0:
        return empty_stats()
    return RegionStats(
        count=int(values.size),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        total=float(np.sum(values)),
    )


def sample_bilinear(channel: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate ``channel`` at the floating-point (xs, ys) coordinates."""
    height, width = channel.shape
    xs = np.clip(xs, 0.0, width - 1.0)
    ys = np.clip(ys, 0.0, height - 1.0)
    x0 = np.floor(xs).astype(np.intp)
    y0 = np.floor(ys).astype(np.intp)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    fx = xs - x0
    fy = ys - y0
    top = channel[y0, x0] * (1.0 - fx) + channel[y0, x1] * fx
    bottom = channel[y1, x0] * (1.0 - fx) + channel[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def line_profile(
    channel: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Intensity sampled along the line (x0, y0)->(x1, y1).

    Returns ``(distances, values)`` where distances are in pixels from the start
    point. ``samples`` defaults to roughly one sample per pixel of line length.
    """
    length = float(np.hypot(x1 - x0, y1 - y0))
    n = samples if samples is not None else int(round(length)) + 1
    n = max(2, int(n))
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    distances = np.linspace(0.0, length, n)
    values = sample_bilinear(np.asarray(channel, dtype=np.float64), xs, ys)
    return distances, values


def pixel_value(channel: np.ndarray, x: float, y: float) -> float | None:
    """Value of the pixel under (x, y), or None if the point is outside the image."""
    height, width = channel.shape
    col = int(np.floor(x))
    row = int(np.floor(y))
    if 0 <= row < height and 0 <= col < width:
        return float(channel[row, col])
    return None
