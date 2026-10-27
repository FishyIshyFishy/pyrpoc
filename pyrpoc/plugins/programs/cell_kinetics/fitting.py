"""Exponential decays fitted to F/F0 traces. Rates are per frame, and ``k``
is always the rate reported: the one the treatment is read from."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from ..building_blocks.parameter_groups.decay_fit import DOUBLE, DOUBLE_FIXED, SINGLE
from .traces import FLOOR, corrected

# The fewest frames each model can be fitted to: one more than its free parameters.
MIN_FRAMES = {SINGLE: 4, DOUBLE: 6, DOUBLE_FIXED: 5}

# Bounds of the original analysis, for traces near 1 at the first frame.
AMPLITUDE = (-2.0, 5.0)
RATE = (0.0, 50.0)
PLATEAU = (-1.0, 3.0)


def exponential(t: np.ndarray, a: float, k: float, c: float) -> np.ndarray:
    return a * np.exp(-k * t) + c


def biexponential(
    t: np.ndarray, a1: float, k1: float, a2: float, k2: float, c: float
) -> np.ndarray:
    return a1 * np.exp(-k1 * t) + a2 * np.exp(-k2 * t) + c


def solve(
    model: Callable[..., np.ndarray],
    t: np.ndarray,
    y: np.ndarray,
    start: list[float],
    bounds: list[tuple[float, float]],
) -> np.ndarray | None:
    """Least-squares parameters from ``start``, or None when the optimizer
    gives up. ``bounds`` is one (low, high) per parameter."""
    lower, upper = (np.array(side) for side in zip(*bounds, strict=True))
    try:
        params, _ = curve_fit(
            model, t, y, p0=np.clip(start, lower, upper), bounds=(lower, upper), maxfev=20000
        )
    except RuntimeError:
        return None
    return params


def fit_single(t: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """``(a, k, c)``, started from the plateau and the first quarter's slope."""
    plateau = float(y.min())
    amplitude = float(y[0]) - plateau
    quarter = max(1, len(t) // 4)
    drop = max(FLOOR, float(y[0] - y[quarter]))
    rate = float(np.clip(drop / ((t[quarter] - t[0]) * max(amplitude, FLOOR)), 0.001, 5.0))
    return solve(exponential, t, y, [amplitude, rate, plateau], [AMPLITUDE, RATE, PLATEAU])


def fit_double(t: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """``(a1, k1, a2, k2, c)`` with ``k1 >= k2``. Started from the single fit
    split in two, off the ridge where both components are equal."""
    single = fit_single(t, y)
    if single is None:
        return None
    a, k, c = single
    bounds = [AMPLITUDE, RATE, AMPLITUDE, RATE, PLATEAU]
    params = solve(biexponential, t, y, [a / 2, k * 2, a / 2, k / 2, c], bounds)
    if params is None:
        return None
    a1, k1, a2, k2, c = params
    # The model is symmetric in its components, so the fast one is put first.
    return params if k1 >= k2 else np.array([a2, k2, a1, k1, c])


def fit_fixed(t: np.ndarray, y: np.ndarray, k1: float) -> np.ndarray | None:
    """``(a1, k1, a2, k2, c)`` with ``k1`` held, so one component is the
    decay imaging alone causes and ``k2`` is the rest."""

    def model(t: np.ndarray, a1: float, a2: float, k2: float, c: float) -> np.ndarray:
        return biexponential(t, a1, k1, a2, k2, c)

    single = fit_single(t, y)
    k, c = (single[1], single[2]) if single is not None else (0.1, 0.0)
    start = [y[0] / 2, y[0] / 2, k, c]
    params = solve(model, t, y, start, [AMPLITUDE, AMPLITUDE, RATE, PLATEAU])
    if params is None:
        return None
    a1, a2, k2, c = params
    return np.array([a1, k1, a2, k2, c])


@dataclass(frozen=True)
class Fit:
    params: np.ndarray
    curve: np.ndarray
    rates: dict[str, float]
    r_squared: float

    @classmethod
    def of(
        cls,
        model: Callable[..., np.ndarray],
        t: np.ndarray,
        data: np.ndarray,
        params: np.ndarray,
        rates: dict[str, float],
    ) -> Fit:
        curve = model(t, *params)
        residual = float(((data - curve) ** 2).sum())
        spread = float(((data - data.mean()) ** 2).sum())
        # A flat trace has no variance to explain.
        r_squared = 1.0 - residual / spread if spread > 0 else float("nan")
        return cls(params, curve, rates, r_squared)

    def describe(self) -> dict[str, Any]:
        """JSON-safe, for the output's metadata."""
        return {
            **{name: float(rate) for name, rate in self.rates.items()},
            "r_squared": self.r_squared,
            "params": [float(value) for value in self.params],
        }


def fit_decay(
    model: str, t: np.ndarray, treated: np.ndarray, control: np.ndarray | None
) -> tuple[np.ndarray, Fit | None]:
    """The trace ``model`` is fitted on, and its fit, None if it did not
    converge. Without a control the treated trace is fitted uncorrected."""
    if model == DOUBLE_FIXED:
        return treated, fit_with_control_rate(t, treated, control)
    data = treated if control is None else corrected(treated, control)
    if model == DOUBLE:
        params = fit_double(t, data)
        if params is None:
            return data, None
        return data, Fit.of(biexponential, t, data, params, {"k": params[3], "k_fast": params[1]})
    params = fit_single(t, data)
    if params is None:
        return data, None
    return data, Fit.of(exponential, t, data, params, {"k": params[1]})


def fit_with_control_rate(
    t: np.ndarray, treated: np.ndarray, control: np.ndarray | None
) -> Fit | None:
    """The uncorrected treated trace, with the control's own rate as one component."""
    if control is None:
        raise ValueError(
            f"'{DOUBLE_FIXED}' needs fluorescent pixels outside the mask to fit the "
            "control's rate; choose another model"
        )
    imaging = fit_single(t, control)
    if imaging is None:
        return None
    params = fit_fixed(t, treated, float(imaging[1]))
    if params is None:
        return None
    return Fit.of(biexponential, t, treated, params, {"k": params[3], "k_imaging": params[1]})
