"""``FocusSurface``: focus as a plane over the stage, fitted to the tiles focused so far, so
each new tile starts its search where focus should be, and searches as far as the plane is
unsure there; and at the end, the plane with focus's drift over the survey taken out."""

# While the survey walks, the plane is fitted to space alone: the next tile is near the last in
# both place and time, so a drift that the plane takes for a tilt still predicts it well. Handed
# on, the plane must hold for the whole region at one time, so the tiles focused again at the end
# are what tell the drift apart from the tilt, and the plane is refitted with both.

from __future__ import annotations

import math

import numpy as np

from pyrpoc.plugins.program_io.inputs import FocusSearch
from pyrpoc.plugins.program_io.outputs import FocusPlane

from ...analysis.plane_fit import DriftingPlane, PlaneFit, fit_drifting_plane, fit_plane

# The search covers focus as surely as this many normal deviations either side would.
SEARCH_DEVIATIONS = 3.0
# Tiles focused again, the fewest that measure the drift with any scatter to spare.
LEAST_REVISITS = 2


class FocusSurface:
    """The z each focused tile settled at and when, the plane through them, and the tiles
    focused again at the end."""

    def __init__(self, start_z_um: float, step_um: float):
        self.start_z_um = start_z_um
        # The search finds focus only to its final step, so no spread is smaller than the
        # rounding to it: a step's uniform error has a deviation of step / sqrt(12).
        self.least_spread = step_um / np.sqrt(12.0)
        self.points: list[tuple[float, float, float]] = []
        self.times_s: list[float] = []
        self.revisits = 0
        self.fitted: PlaneFit | None = None
        self.plane = FocusPlane(start_z_um, 0.0, 0.0, 0.0, 0)

    def add(self, x_um: float, y_um: float, z_um: float, time_s: float) -> None:
        """A tile focused as the survey walks, which the plane is fitted to at once."""
        self.points.append((x_um, y_um, z_um))
        self.times_s.append(time_s)
        fitted = fit_plane(np.asarray(self.points), self.least_spread)
        self.fitted = fitted
        self.plane = FocusPlane(fitted.z0, fitted.tilt_x, fitted.tilt_y, fitted.spread, fitted.kept)

    def add_revisit(self, x_um: float, y_um: float, z_um: float, time_s: float) -> None:
        """A tile focused again at the end, which only the final plane is fitted to."""
        self.points.append((x_um, y_um, z_um))
        self.times_s.append(time_s)
        self.revisits += 1

    def settled(self, at_s: float) -> tuple[FocusPlane, DriftingPlane | None]:
        """The plane as it stood at ``at_s`` and the drift fitted with it, when enough tiles
        were focused again to tell the two apart; else the plane as walked, and None."""
        if self.revisits < LEAST_REVISITS:
            return self.plane, None
        points, times = np.asarray(self.points), np.asarray(self.times_s)
        drifting = fit_drifting_plane(points, times, at_s, self.least_spread)
        fitted = drifting.plane
        plane = FocusPlane(fitted.z0, fitted.tilt_x, fitted.tilt_y, fitted.spread, fitted.kept)
        return plane, drifting

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
