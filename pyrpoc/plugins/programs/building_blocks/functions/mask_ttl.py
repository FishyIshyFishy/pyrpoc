"""Bound masks as the per-sample TTL a galvo raster plays on the digital lines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..parameter_groups.modulation import Mask
from ..parameter_groups.scan import ScanGroup


def mask_ttl(
    masks: Sequence[Mask], scan: ScanGroup, samples_per_pixel: int, device_name: str
) -> dict[str, np.ndarray]:
    """One flat boolean TTL signal per bound digital line, a sample for every
    raster sample. A mask that is all zero on the scan grid gets no line, so no
    DO task is created for it."""
    ttl: dict[str, np.ndarray] = {}
    for mask in masks:
        lit = np.zeros((scan.y_pixels, scan.total_x), dtype=bool)
        lit[:, scan.kept_columns] = mask.on_grid(scan.y_pixels, scan.x_pixels)
        if np.any(lit):
            ttl[mask.channel(device_name)] = np.repeat(lit, samples_per_pixel, axis=1).reshape(-1)
    return ttl
