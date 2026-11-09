"""``SurveyMap``: what a loop takes from the survey it starts from."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.plugins.program_io.outputs import FocusDrift, FocusPlane, MotionRate, Unit

from ...interfaces.overview import Overview


@dataclass(frozen=True)
class Visit:
    """Where the survey centred one tile, and when it imaged it."""

    x_um: float
    y_um: float
    time_s: float


@dataclass(frozen=True)
class SurveyMap:
    """The dish as the survey found it: its focus plane and how fast focus drifted, how fast
    things move, its overview, and where and when each tile was imaged."""

    focus: FocusPlane
    drift: FocusDrift | None
    motion: MotionRate | None
    overview: Overview
    visits: tuple[Visit, ...]

    @property
    def focused_s(self) -> float:
        """When the survey imaged its last tile, the time its focus plane holds for."""
        return max(visit.time_s for visit in self.visits)

    def seen_at(self, unit: Unit) -> float:
        """When the survey imaged ``unit``: the time of the tile nearest its centre."""
        cx, cy = unit.centre
        distances = [(v.x_um - cx) ** 2 + (v.y_um - cy) ** 2 for v in self.visits]
        return self.visits[int(np.argmin(distances))].time_s
