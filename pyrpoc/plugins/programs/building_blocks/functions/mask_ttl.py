"""Drawn masks as the per-sample TTL a galvo raster plays on their digital lines."""

from __future__ import annotations

import numpy as np

from pyrpoc.plugins.program_io.inputs import Mask, MaskLine, ScanGeometry


def mask_ttl(
    masks: tuple[Mask, ...],
    lines: tuple[MaskLine, ...],
    scan: ScanGeometry,
    samples_per_pixel: int,
    device_name: str,
) -> dict[str, np.ndarray]:
    """One flat boolean TTL signal per mask's digital line, ``lines`` in the masks' order, a
    sample for every raster sample."""
    # A switched-off mask drives nothing, and one that is all zero on the scan grid gets no
    # line, so no DO task is created for either.
    ttl: dict[str, np.ndarray] = {}
    for mask, line in zip(masks, lines, strict=True):
        if not line.enabled:
            continue
        lit = np.zeros((scan.y_pixels, scan.total_x), dtype=bool)
        lit[:, scan.kept_columns] = mask.on_grid(scan.y_pixels, scan.x_pixels)
        if np.any(lit):
            ttl[line.channel(device_name)] = np.repeat(lit, samples_per_pixel, axis=1).reshape(-1)
    return ttl
