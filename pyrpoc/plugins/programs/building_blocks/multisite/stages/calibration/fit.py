"""The line through measured image shifts against stage moves, and when it is not one."""

# The line is fitted with an offset as well as a slope. Wherever the stage lands short of or past
# its targets by the same amount every time, every shift is off by that amount, whichever way the
# move went; that says nothing about the scale, and a line forced through zero would tilt to
# absorb it.
#
# Whether the fit is good enough is judged by how precisely it knows the slope, which is what the
# calibration is for, rather than by its scatter against a share of the move: a stage that lands
# within a fraction of a µm can fail any fixed share of a short move and still give a slope known
# to a fraction of a percent.

from __future__ import annotations

import numpy as np

from ...analysis.noise import MAD_TO_SIGMA

# A point further than this many robust deviations from the fit is a mismatch, not a measure.
OUTLIER_DEVIATIONS = 5.0
# A slope known no better than this fraction of itself did not measure the scale.
WORST_UNCERTAINTY = 0.01
# Two parameters, and at least two points to spare to measure the scatter about them.
LEAST_KEPT = 4


class CalibrationError(Exception):
    """The calibration could not be measured."""


def fit(moves: np.ndarray, shifts: np.ndarray) -> tuple[float, float]:
    """The slope of shift against move, with an offset, refitted without mismatched points,
    and the scatter of the points kept about it in µm of move."""
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
    if uncertainty > WORST_UNCERTAINTY or keep.sum() < len(moves) // 2:
        raise CalibrationError(
            f"the measured shifts scatter too much about a line (the scale is uncertain by "
            f"{uncertainty:.1%}): try another spot"
        )
    return slope, scatter / abs(slope)
