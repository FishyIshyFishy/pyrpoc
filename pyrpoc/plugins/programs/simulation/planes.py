"""The test patterns a simulated channel can show. Every plane function takes
the same arguments so ``PLANES`` can dispatch on the pattern's name."""

from __future__ import annotations

import numpy as np

from ..building_blocks.functions.synthetic_detector import seeded_rng

# Blobs per channel in the "cells" pattern.
BLOB_COUNT = 14


def axes(y_pixels: int, x_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """Column and row vectors, so plane maths broadcasts to ``(H, W)``."""
    ys = np.arange(y_pixels, dtype=np.float32)[:, None]
    xs = np.arange(x_pixels, dtype=np.float32)[None, :]
    return ys, xs


def cells_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """Gaussian blobs drifting across the field, each channel its own set.

    Separable exponentials keep a 512x512 frame at milliseconds; distances wrap
    so a long run never empties the field.
    """
    generator = seeded_rng(seed, channel, 0xB10B)
    centre_y = generator.uniform(0.0, y_pixels, BLOB_COUNT)
    centre_x = generator.uniform(0.0, x_pixels, BLOB_COUNT)
    extent = max(y_pixels, x_pixels)
    sigma = generator.uniform(0.02, 0.055, BLOB_COUNT) * extent
    amplitude = generator.uniform(0.4, 1.0, BLOB_COUNT)
    heading = generator.uniform(0.0, 2.0 * np.pi, BLOB_COUNT)

    travelled = drift * frame_index
    centre_y = (centre_y + np.sin(heading) * travelled) % y_pixels
    centre_x = (centre_x + np.cos(heading) * travelled) % x_pixels

    ys, xs = axes(y_pixels, x_pixels)
    plane = np.zeros((y_pixels, x_pixels), dtype=np.float32)
    for index in range(BLOB_COUNT):
        spread = 2.0 * sigma[index] ** 2
        dy = np.abs(ys - centre_y[index])
        dy = np.minimum(dy, y_pixels - dy)
        dx = np.abs(xs - centre_x[index])
        dx = np.minimum(dx, x_pixels - dx)
        plane += amplitude[index] * np.exp(-(dy**2) / spread) * np.exp(-(dx**2) / spread)
    return np.clip(plane, 0.0, 1.0)


def rings_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """Concentric sine rings, breathing outwards. Good for spotting resampling."""
    del seed
    ys, xs = axes(y_pixels, x_pixels)
    radius = np.sqrt((ys - y_pixels / 2.0) ** 2 + (xs - x_pixels / 2.0) ** 2)
    period = max(4.0, min(y_pixels, x_pixels) / 12.0)
    phase = channel * (np.pi / 3.0) + drift * frame_index * 0.25
    return 0.5 * (1.0 + np.sin(2.0 * np.pi * radius / period - phase)).astype(np.float32)


def gradient_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """A linear ramp whose direction differs per channel. Shows orientation.
    It sweeps as a triangle wave, so the drift never puts a seam across the frame."""
    del seed
    ys, xs = axes(y_pixels, x_pixels)
    angle = channel * (np.pi / 3.0)
    ramp = np.cos(angle) * (xs / max(1, x_pixels - 1)) + np.sin(angle) * (ys / max(1, y_pixels - 1))
    shift = drift * frame_index / max(y_pixels, x_pixels)
    swept = (ramp + shift) % 2.0
    return np.abs(1.0 - swept).astype(np.float32)


def checkerboard_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """Hard-edged squares that march diagonally. Shows pixel alignment."""
    del seed
    square = max(2, min(y_pixels, x_pixels) // 8)
    offset = int(round(drift * frame_index)) + channel * (square // 2)
    ys, xs = axes(y_pixels, x_pixels)
    board = (((ys + offset) // square) + ((xs + offset) // square)) % 2.0
    return np.broadcast_to(board, (y_pixels, x_pixels)).astype(np.float32)


def flat_plane(
    y_pixels: int, x_pixels: int, *, channel: int, seed: int, drift: float, frame_index: int
) -> np.ndarray:
    """A uniform field. With noise turned up it is a pure noise source."""
    del channel, seed, drift, frame_index
    return np.full((y_pixels, x_pixels), 0.5, dtype=np.float32)


# Keyed by ``PATTERNS``, the choices ``SignalGroup.pattern`` offers.
PLANES = {
    "cells": cells_plane,
    "rings": rings_plane,
    "gradient": gradient_plane,
    "checkerboard": checkerboard_plane,
    "flat": flat_plane,
}
