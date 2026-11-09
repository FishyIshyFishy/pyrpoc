"""A plane fitted robustly to points: least squares, refitted without the points far from it,
and how far from it a new point may lie."""

# Fitted about the points' own centre, so the height and the tilts are measured independently;
# about the stage's origin, thousands of µm away, a level plane and a tilted one can fit equally
# well and the solver may pick either.
#
# A tilt is fitted only along the directions the points spread across. Points on one row of
# tiles say nothing about the tilt across rows, and the plane must not pretend to know it.
#
# How far a new point may lie from the plane is the least-squares prediction interval: the
# points' scatter about the plane, the plane's own uncertainty there (growing with distance from
# the points), and a Student t quantile, so a plane fitted to few points is trusted less. Where
# the points say nothing (too few to measure their scatter, or off the line they lie on) it is
# infinite.
#
# Points taken over time while focus drifts can be fitted with the drift as one more term,
# linear in time. A survey walks its rows in order, so time and y rise together and the drift
# could pass for a tilt in y; only points taken again at the same place later tell them apart.

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .noise import MAD_TO_SIGMA

# A point this many robust deviations from the plane is a measurement that went wrong.
OUTLIER_DEVIATIONS = 3.0
# Points spread less than this across a direction say nothing about the tilt along it.
SPAN_UM = 1.0


@dataclass(frozen=True)
class PlaneFit:
    """z = z0 + tilt_x * x + tilt_y * y, the spread of the points kept about it, and how many
    were kept; and what a prediction from it needs to know how far to trust itself."""

    z0: float
    tilt_x: float
    tilt_y: float
    spread: float
    kept: int
    centre: tuple[float, float]
    # Unit vectors of the directions the kept points spread across, one per row.
    spanned: np.ndarray
    # The inverse of the normal matrix of the fit, in the centred, spanned coordinates.
    inverse: np.ndarray
    # The kept points less the parameters fitted; below 1 the scatter is unmeasured.
    freedom: int

    def z_at(self, x_um: float, y_um: float) -> float:
        return self.z0 + self.tilt_x * x_um + self.tilt_y * y_um

    def reach(self, x_um: float, y_um: float, deviations: float) -> float:
        """Half the width of the interval about the plane at (``x_um``, ``y_um``) that holds
        a new point as often as ``deviations`` normal deviations would; inf where the points
        say nothing."""
        if self.freedom < 1:
            return math.inf
        offset = np.array([x_um - self.centre[0], y_um - self.centre[1]])
        along = self.spanned @ offset
        off_line = offset - self.spanned.T @ along
        if float(np.hypot(*off_line)) > SPAN_UM:
            return math.inf
        basis = np.concatenate([[1.0], along])
        leverage = float(basis @ self.inverse @ basis)
        quantile = student_t_quantile(normal_coverage(deviations), self.freedom)
        return quantile * self.spread * math.sqrt(1.0 + leverage)


def normal_coverage(deviations: float) -> float:
    """The one-sided probability a normal value lies below ``deviations``."""
    return 0.5 * (1.0 + math.erf(deviations / math.sqrt(2.0)))


def student_t_quantile(probability: float, freedom: int) -> float:
    """The value a Student t with ``freedom`` degrees lies below with ``probability``."""
    # Bisection on the distribution, integrated in u = atan(t), where even its heavy tail
    # becomes a smooth curve over a finite range.
    log_scale = (
        math.lgamma((freedom + 1) / 2.0)
        - math.lgamma(freedom / 2.0)
        - 0.5 * math.log(freedom * math.pi)
    )

    def below(t: float) -> float:
        u = np.linspace(0.0, math.atan(t), 4001)
        tangent = np.tan(u)
        density = np.exp(log_scale) * (1.0 + tangent**2 / freedom) ** (-(freedom + 1) / 2.0)
        integrand = density / np.cos(u) ** 2
        return 0.5 + float(np.sum((integrand[1:] + integrand[:-1]) * np.diff(u)) / 2.0)

    low, high = 0.0, 1.0
    while below(high) < probability:
        low, high = high, high * 2.0
    for _ in range(60):
        middle = (low + high) / 2.0
        low, high = (middle, high) if below(middle) < probability else (low, middle)
    return (low + high) / 2.0


def spanned_directions(xy: np.ndarray) -> np.ndarray:
    """Unit vectors, one per row, of the directions centred points ``xy`` spread across."""
    if len(xy) < 2:
        return np.zeros((0, 2))
    _, values, directions = np.linalg.svd(xy, full_matrices=False)
    spread = values / math.sqrt(len(xy))
    return directions[spread > SPAN_UM]


@dataclass(frozen=True)
class RobustFit:
    """Least squares refitted without the points far from it: the coefficients, how many points
    were kept, their spread about the fit, the inverse of their normal matrix, and the kept
    points less the parameters."""

    coefficients: np.ndarray
    kept: int
    spread: float
    inverse: np.ndarray
    freedom: int


def robust_fit(design: np.ndarray, z: np.ndarray, least_spread: float) -> RobustFit:
    """The least-squares fit of ``z`` to the columns of ``design``, refitted without the points
    far from it, never claiming a spread below ``least_spread``."""
    parameters = design.shape[1]
    keep = np.ones(len(z), dtype=bool)
    for _ in range(2):
        coefficients, *_ = np.linalg.lstsq(design[keep], z[keep], rcond=None)
        errors = z - design @ coefficients
        spread = max(float(np.median(np.abs(errors[keep]))) * MAD_TO_SIGMA, least_spread)
        keep = np.abs(errors) <= OUTLIER_DEVIATIONS * spread
        # Rejecting points needs some to spare: below that, none can be told to be wrong.
        if keep.sum() <= parameters:
            keep = np.ones(len(z), dtype=bool)
    coefficients, *_ = np.linalg.lstsq(design[keep], z[keep], rcond=None)
    freedom = int(keep.sum()) - parameters
    errors = z[keep] - design[keep] @ coefficients
    # The scatter that the fit's own parameters have not already soaked up.
    scatter = math.sqrt(float(np.sum(errors**2)) / freedom) if freedom >= 1 else 0.0
    inverse = np.linalg.pinv(design[keep].T @ design[keep])
    return RobustFit(coefficients, int(keep.sum()), max(scatter, least_spread), inverse, freedom)


def spatial_design(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The centre of points ``xy``, the directions they spread across, and the design of a
    plane over them: a constant, then each spanned direction about the centre."""
    centre = xy.mean(axis=0)
    spanned = spanned_directions(xy - centre)
    design = np.column_stack([np.ones(len(xy)), (xy - centre) @ spanned.T])
    return centre, spanned, design


def plane_from(
    fit: RobustFit, centre: np.ndarray, spanned: np.ndarray, inverse: np.ndarray
) -> PlaneFit:
    """The plane in stage coordinates from the leading, spatial coefficients of ``fit``, whose
    normal matrix for them inverts to ``inverse``."""
    tilt = spanned.T @ fit.coefficients[1 : 1 + len(spanned)]
    z0 = float(fit.coefficients[0] - tilt @ centre)
    return PlaneFit(
        z0,
        float(tilt[0]),
        float(tilt[1]),
        fit.spread,
        fit.kept,
        (float(centre[0]), float(centre[1])),
        spanned,
        inverse,
        fit.freedom,
    )


def fit_plane(points: np.ndarray, least_spread: float) -> PlaneFit:
    """The plane through ``points``, an ``(N, 3)`` array of (x, y, z) with N at least 1, never
    claiming a spread below ``least_spread``."""
    centre, spanned, design = spatial_design(points[:, :2])
    fit = robust_fit(design, points[:, 2], least_spread)
    return plane_from(fit, centre, spanned, fit.inverse)


@dataclass(frozen=True)
class DriftingPlane:
    """A plane fitted to points taken while focus drifted steadily: the plane as it stood at the
    fit's reference time, and the drift's rate and its standard error, in µm per second."""

    plane: PlaneFit
    rate: float
    rate_error: float


def fit_drifting_plane(
    points: np.ndarray, times_s: np.ndarray, at_s: float, least_spread: float
) -> DriftingPlane:
    """The plane through ``points``, an ``(N, 3)`` array of (x, y, z) taken at ``times_s``, with
    z drifting linearly in time, as it stood at ``at_s``."""
    centre, spanned, spatial = spatial_design(points[:, :2])
    # In minutes, so the drift's column is of a size with the others.
    elapsed_min = (times_s - at_s) / 60.0
    fit = robust_fit(np.column_stack([spatial, elapsed_min]), points[:, 2], least_spread)
    # The spatial block of the inverse is the plane's covariance at ``at_s``, its drift's
    # uncertainty included.
    terms = spatial.shape[1]
    plane = plane_from(fit, centre, spanned, fit.inverse[:terms, :terms])
    rate_error = fit.spread * math.sqrt(float(fit.inverse[-1, -1]))
    return DriftingPlane(plane, float(fit.coefficients[-1]) / 60.0, rate_error / 60.0)
