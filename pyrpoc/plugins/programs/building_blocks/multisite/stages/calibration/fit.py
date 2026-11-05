"""The line through measured image shifts against stage moves, and when it is not one."""

from __future__ import annotations

import numpy as np

# A point further than this many median deviations from the fit is a mismatch, not a measure.
OUTLIER_DEVIATIONS = 5.0
# A fit whose residual is more than this fraction of the largest move did not measure a line.
WORST_RESIDUAL = 0.02


class CalibrationError(Exception):
    """The calibration could not be measured."""


def fit(moves: np.ndarray, shifts: np.ndarray) -> tuple[float, float]:
    """The slope through the origin of shift against move, refitted without mismatched
    points, and the residual in µm."""
    keep = np.ones(len(moves), dtype=bool)
    for _ in range(2):
        slope = float(np.sum(moves[keep] * shifts[keep]) / np.sum(moves[keep] ** 2))
        errors = np.abs(shifts - slope * moves)
        spread = float(np.median(errors[keep])) * 1.4826 + 1e-6
        keep = errors <= OUTLIER_DEVIATIONS * spread
    slope = float(np.sum(moves[keep] * shifts[keep]) / np.sum(moves[keep] ** 2))
    residual = float(np.sqrt(np.mean((shifts[keep] - slope * moves[keep]) ** 2))) / abs(slope)
    if residual > WORST_RESIDUAL * float(np.max(np.abs(moves))) or keep.sum() < len(moves) // 2:
        raise CalibrationError("the measured shifts do not fall on a line: try another spot")
    return slope, residual
