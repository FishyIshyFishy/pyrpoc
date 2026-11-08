"""``Phase``: which part of a scan the loop is in."""

from __future__ import annotations

from enum import Enum

from pyrpoc.plugins.program_io.outputs import FocusingAtUnit, LoopActivity, MovingToUnit


class Phase(Enum):
    MOVING = "Moving"
    FOCUSING = "Focusing"
    SCANNING = "Scanning"


def phase_of(activity: LoopActivity) -> Phase:
    match activity:
        case MovingToUnit():
            return Phase.MOVING
        case FocusingAtUnit():
            return Phase.FOCUSING
        case _:
            return Phase.SCANNING
