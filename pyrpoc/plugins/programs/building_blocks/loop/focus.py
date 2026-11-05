"""Keeping focus through a loop: the survey's focus plane, plus an offset that periodic checks
on the units keep up to date as focus drifts."""

from __future__ import annotations

import math
from collections.abc import Callable

from pyrpoc.plugins.program_io.inputs import FocusSearch
from pyrpoc.plugins.program_io.outputs import FocusPlane

from ..focus.climb import climb

# How many of the plane's own deviations a check searches either side of its prediction.
SEARCH_DEVIATIONS = 3.0


class FocusKeeper:
    """Where focus is: the survey's plane, never refitted, shifted by what the last check
    found."""

    def __init__(self, plane: FocusPlane, every_s: float, final_step_um: float):
        self.plane = plane
        self.every_s = every_s
        self.final_step_um = final_step_um
        self.offset_um = 0.0
        self.checked_s = -math.inf

    def z_at(self, x_um: float, y_um: float) -> float:
        return self.plane.z_at(x_um, y_um) + self.offset_um

    def due(self, now_s: float) -> bool:
        return now_s - self.checked_s >= self.every_s

    def search(self) -> FocusSearch:
        """Around the prediction, as far as the plane's spread and the climb's step allow."""
        span = max(SEARCH_DEVIATIONS * self.plane.residual_um, 4.0 * self.final_step_um)
        return FocusSearch(span, span / 2.0, self.final_step_um)

    def check(
        self, measure: Callable[[float], float], x_um: float, y_um: float, now_s: float
    ) -> float:
        """Find focus at (``x_um``, ``y_um``), scoring each z with ``measure``, and take the
        offset from it; returns the z found."""
        best = climb(measure, lambda _step: None, self.z_at(x_um, y_um), self.search())
        self.offset_um = best.best_z_um - self.plane.z_at(x_um, y_um)
        self.checked_s = now_s
        return best.best_z_um
