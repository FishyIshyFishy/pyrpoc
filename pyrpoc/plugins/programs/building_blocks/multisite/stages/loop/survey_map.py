"""``SurveyMap``: what a loop takes from the survey it starts from."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.plugins.program_io.inputs import Unit
from pyrpoc.plugins.program_io.outputs import FocusPlane, MotionRate

from ...interfaces.overview import Overview


@dataclass(frozen=True)
class Visit:
    """Where the survey centred one tile, and when it imaged it."""

    x_um: float
    y_um: float
    time_s: float


@dataclass(frozen=True)
class SurveyMap:
    """The dish as the survey found it: its focus plane, how fast things move, its overview,
    and where and when each tile was imaged."""

    focus: FocusPlane
    motion: MotionRate | None
    overview: Overview
    visits: tuple[Visit, ...]

    def seen_at(self, unit: Unit) -> float:
        """When the survey imaged ``unit``: the time of the tile nearest its centre."""
        cx, cy = unit.centre
        distances = [(v.x_um - cx) ** 2 + (v.y_um - cy) ** 2 for v in self.visits]
        return self.visits[int(np.argmin(distances))].time_s
