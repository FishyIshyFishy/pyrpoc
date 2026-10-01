"""One endless synthetic slide, as a function of absolute pixel position.

Neighbouring tiles share the features in their overlap exactly as a real
specimen would. Blobs are seeded per lattice cell rather than drawn on one
canvas, so a wide grid costs no more memory than one tile.
"""

from __future__ import annotations

import numpy as np

from ..building_blocks.functions.synthetic_detector import seeded_rng

# The slide's lattice: each cell holds its own blobs, drawn from its own seed.
CELL_PIXELS = 48
BLOBS_PER_CELL = 3
SIGMA_PIXELS = (1.5, 6.0)
# A blob is negligible beyond this many sigmas, so cells this far out are skipped.
REACH_SIGMAS = 3.0


def add_cell_blobs(
    plane: np.ndarray, top: int, left: int, *, cell_y: int, cell_x: int, seed: int
) -> None:
    """Add one lattice cell's blobs to ``plane``, whose pixel (0, 0) is slide
    pixel (``top``, ``left``). Each blob only touches the window it reaches."""
    generator = seeded_rng(seed, cell_y, cell_x)
    centre_y = cell_y * CELL_PIXELS + generator.uniform(0.0, CELL_PIXELS, BLOBS_PER_CELL)
    centre_x = cell_x * CELL_PIXELS + generator.uniform(0.0, CELL_PIXELS, BLOBS_PER_CELL)
    sigma = generator.uniform(*SIGMA_PIXELS, BLOBS_PER_CELL)
    amplitude = generator.uniform(0.3, 1.0, BLOBS_PER_CELL)

    height, width = plane.shape
    for cy, cx, s, a in zip(centre_y, centre_x, sigma, amplitude, strict=True):
        reach = REACH_SIGMAS * s
        y0, y1 = max(0, int(cy - reach) - top), min(height, int(cy + reach) + 1 - top)
        x0, x1 = max(0, int(cx - reach) - left), min(width, int(cx + reach) + 1 - left)
        if y0 >= y1 or x0 >= x1:
            continue
        ys = np.arange(y0 + top, y1 + top, dtype=np.float32) - cy
        xs = np.arange(x0 + left, x1 + left, dtype=np.float32) - cx
        spread = 2.0 * s**2
        plane[y0:y1, x0:x1] += a * np.outer(np.exp(-(ys**2) / spread), np.exp(-(xs**2) / spread))


def slide_patch(top: int, left: int, height: int, width: int, *, seed: int) -> np.ndarray:
    """The slide's ``(height, width)`` window starting at pixel (``top``, ``left``)."""
    margin = int(REACH_SIGMAS * SIGMA_PIXELS[1]) + 1
    plane = np.zeros((height, width), dtype=np.float32)
    for cell_y in range((top - margin) // CELL_PIXELS, (top + height + margin) // CELL_PIXELS + 1):
        for cell_x in range(
            (left - margin) // CELL_PIXELS, (left + width + margin) // CELL_PIXELS + 1
        ):
            add_cell_blobs(plane, top, left, cell_y=cell_y, cell_x=cell_x, seed=seed)
    return np.clip(plane, 0.0, 1.0)
