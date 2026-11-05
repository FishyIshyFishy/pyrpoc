"""``FocusSurface``: focus as a plane over the stage, fitted to the tiles focused so far, so
each new tile starts its search where focus should be."""

from __future__ import annotations

import numpy as np

from pyrpoc.plugins.program_io.inputs import FocusSearch
from pyrpoc.plugins.program_io.outputs import FocusPlane

from ...analysis.plane_fit import PLANE_POINTS, fit_plane

# How many deviations of the plane's own spread the search must cover either side.
SEARCH_DEVIATIONS = 3.0


class FocusSurface:
    """The z each focused tile settled at, and the plane through them."""

    def __init__(self, start_z_um: float, step_um: float):
        self.start_z_um = start_z_um
        # The search finds focus only to its final step, so no spread is smaller than the
        # rounding to it: a step's uniform error has a deviation of step / sqrt(12).
        self.least_spread = step_um / np.sqrt(12.0)
        self.points: list[tuple[float, float, float]] = []
        self.plane = FocusPlane(start_z_um, 0.0, 0.0, 0.0, 0)

    def add(self, x_um: float, y_um: float, z_um: float) -> None:
        self.points.append((x_um, y_um, z_um))
        self.plane = self.fit()

    def fit(self) -> FocusPlane:
        points = np.asarray(self.points)
        # Until a plane can be fitted, the last focus found is the best guess.
        if len(points) < PLANE_POINTS:
            return FocusPlane(float(points[-1, 2]), 0.0, 0.0, 0.0, len(points))
        plane = fit_plane(points, self.least_spread)
        return FocusPlane(plane.z0, plane.tilt_x, plane.tilt_y, plane.spread, plane.kept)

    def predict(self, x_um: float, y_um: float) -> float:
        return self.plane.z_at(x_um, y_um)

    def search(self, full: FocusSearch) -> FocusSearch:
        """How far to search around a prediction: the whole range until the plane is fitted,
        then as far as the plane's spread says focus can be from it."""
        if self.plane.points < PLANE_POINTS:
            return full
        reach = max(SEARCH_DEVIATIONS * self.plane.residual_um, 2.0 * full.final_step_um)
        span = min(full.search_range_um, reach)
        return FocusSearch(span, min(full.initial_step_um, span / 2.0), full.final_step_um)
