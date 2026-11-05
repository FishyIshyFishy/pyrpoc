"""A plane fitted robustly to points: least squares, refitted without the points far from it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .noise import MAD_TO_SIGMA

# A plane needs three points.
PLANE_POINTS = 3
# A point this many robust deviations from the plane is a measurement that went wrong.
OUTLIER_DEVIATIONS = 3.0


@dataclass(frozen=True)
class PlaneFit:
    """z = z0 + tilt_x * x + tilt_y * y, the spread of the points kept about it, and how many
    were kept."""

    z0: float
    tilt_x: float
    tilt_y: float
    spread: float
    kept: int


def fit_plane(points: np.ndarray, least_spread: float) -> PlaneFit:
    """The plane through ``points``, an ``(N, 3)`` array of (x, y, z) with N at least
    ``PLANE_POINTS``, never claiming a spread below ``least_spread``."""
    design = np.column_stack([np.ones(len(points)), points[:, 0], points[:, 1]])
    keep = np.ones(len(points), dtype=bool)
    for _ in range(2):
        coefficients, *_ = np.linalg.lstsq(design[keep], points[keep, 2], rcond=None)
        errors = points[:, 2] - design @ coefficients
        spread = max(float(np.median(np.abs(errors[keep]))) * MAD_TO_SIGMA, least_spread)
        keep = np.abs(errors) <= OUTLIER_DEVIATIONS * spread
        if keep.sum() < PLANE_POINTS:
            keep = np.ones(len(points), dtype=bool)
    z0, tilt_x, tilt_y = (float(value) for value in coefficients)
    return PlaneFit(z0, tilt_x, tilt_y, spread, int(keep.sum()))
