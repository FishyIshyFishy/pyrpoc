"""``FocusSurface``: focus as a plane over the stage, fitted to the tiles focused so far, so
each new tile starts its search where focus should be, and searches as far as the plane is
unsure there."""

from __future__ import annotations

import math

import numpy as np

from pyrpoc.plugins.program_io.inputs import FocusSearch
from pyrpoc.plugins.program_io.outputs import FocusPlane

from ...analysis.plane_fit import PlaneFit, fit_plane

# The search covers focus as surely as this many normal deviations either side would.
SEARCH_DEVIATIONS = 3.0


class FocusSurface:
    """The z each focused tile settled at, and the plane through them."""

    def __init__(self, start_z_um: float, step_um: float):
        self.start_z_um = start_z_um
        # The search finds focus only to its final step, so no spread is smaller than the
        # rounding to it: a step's uniform error has a deviation of step / sqrt(12).
        self.least_spread = step_um / np.sqrt(12.0)
        self.points: list[tuple[float, float, float]] = []
        self.fitted: PlaneFit | None = None
        self.plane = FocusPlane(start_z_um, 0.0, 0.0, 0.0, 0)

    def add(self, x_um: float, y_um: float, z_um: float) -> None:
        self.points.append((x_um, y_um, z_um))
        fitted = fit_plane(np.asarray(self.points), self.least_spread)
        self.fitted = fitted
        self.plane = FocusPlane(fitted.z0, fitted.tilt_x, fitted.tilt_y, fitted.spread, fitted.kept)

    def predict(self, x_um: float, y_um: float) -> float:
        return self.plane.z_at(x_um, y_um)

    def search(self, full: FocusSearch, x_um: float, y_um: float) -> FocusSearch:
        """How far to search around the prediction at (``x_um``, ``y_um``): as far as focus can
        be from it there, and the whole range where the plane cannot say."""
        if self.fitted is None:
            return full
        reach = self.fitted.reach(x_um, y_um, SEARCH_DEVIATIONS)
        if not math.isfinite(reach) or reach >= full.search_range_um:
            return full
        span = max(reach, 2.0 * full.final_step_um)
        return FocusSearch(span, min(full.initial_step_um, span / 2.0), full.final_step_um)
