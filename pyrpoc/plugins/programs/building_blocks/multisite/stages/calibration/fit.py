"""The line through measured image shifts against stage moves, and when it is not one."""

# The line is fitted with an offset as well as a slope. Wherever the stage lands short of or past
# its targets by the same amount every time, every shift is off by that amount, whichever way the
# move went; that says nothing about the scale, and a line forced through zero would tilt to
# absorb it.
#
# How good the fit is, is judged by how precisely it knows the slope, which is what the
# calibration is for, rather than by its scatter against a share of the move: a stage that lands
# within a fraction of a µm can fail any fixed share of a short move and still give a slope known
# to a fraction of a percent.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...analysis.noise import MAD_TO_SIGMA

# A point further than this many robust deviations from the fit is a mismatch, not a measure.
OUTLIER_DEVIATIONS = 5.0
# Two parameters, and at least two points to spare to measure the scatter about them.
LEAST_KEPT = 4


class CalibrationError(Exception):
    """The calibration could not be measured."""


@dataclass(frozen=True)
class Line:
    """Shift against move: the slope, the scatter of the points kept about it in µm of move,
    the slope's standard error as a fraction of it, and which points were kept."""

    slope: float
    scatter_um: float
    uncertainty: float
    kept: np.ndarray


def fit(moves: np.ndarray, shifts: np.ndarray) -> Line:
    """The line of shift against move, with an offset, refitted without mismatched points."""
    design = np.column_stack([moves, np.ones_like(moves)])
    # Started from medians, a mismatch can't drag the first line and hide among the errors.
    slope = float(np.median(shifts / moves))
    coefficients = np.array([slope, float(np.median(shifts - slope * moves))])
    keep = np.ones(len(moves), dtype=bool)
    for _ in range(2):
        errors = np.abs(shifts - design @ coefficients)
        spread = float(np.median(errors[keep])) * MAD_TO_SIGMA + 1e-6
        keep = errors <= OUTLIER_DEVIATIONS * spread
        if keep.sum() < LEAST_KEPT:
            raise CalibrationError(
                "too few of the moves agree on a line: the stage or the sample moved "
                "unpredictably; try another spot"
            )
        coefficients, *_ = np.linalg.lstsq(design[keep], shifts[keep], rcond=None)
    slope = float(coefficients[0])
    errors = shifts[keep] - design[keep] @ coefficients
    scatter = float(np.sqrt(np.sum(errors**2) / (keep.sum() - 2)))
    inverse = np.linalg.inv(design[keep].T @ design[keep])
    uncertainty = scatter * float(np.sqrt(inverse[0, 0])) / abs(slope)
    return Line(slope, scatter / abs(slope), uncertainty, keep)


def cross_slope(moves: np.ndarray, crosses: np.ndarray, kept: np.ndarray) -> float:
    """The slope, with an offset, of the shifts across the axis against the moves along it,
    over the points ``kept``."""
    design = np.column_stack([moves, np.ones_like(moves)])[kept]
    coefficients, *_ = np.linalg.lstsq(design, crosses[kept], rcond=None)
    return float(coefficients[0])
