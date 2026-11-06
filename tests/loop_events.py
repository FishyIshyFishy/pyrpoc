"""Loop events for tests: a loop on two sites, written as the loop would log them."""

from __future__ import annotations

from pyrpoc.plugins.program_io.outputs import (
    LoopEvent,
    LoopStarted,
    Measured,
    StepDue,
    UnitJoined,
    UnitScanned,
)

SQUARE = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))


def scanned(unit: int, time_s: float, found: bool) -> UnitScanned:
    return UnitScanned(time_s, unit, "monitor", 0, 0.0, 5.0, 5.0, 0.0, (0.0, 0.0), found, 20.0, ())


def two_sites() -> list[LoopEvent]:
    """A loop started at 100 s that added sites 1 and 2 and measured each once."""
    return [
        LoopStarted(100.0, ("ch0",), ("monitor",)),
        UnitJoined(100.0, 1, "key-1", "treated", SQUARE),
        UnitJoined(100.0, 2, "key-2", "", SQUARE),
        StepDue(100.0, 1, "monitor", 100.0),
        StepDue(100.0, 2, "monitor", 100.0),
        scanned(1, 110.0, True),
        Measured(110.0, 1, "monitor", "intensity", (1.0,)),
        StepDue(110.0, 1, "monitor", 170.0),
        scanned(2, 120.0, True),
        Measured(120.0, 2, "monitor", "intensity", (2.0,)),
        StepDue(120.0, 2, "monitor", 180.0),
    ]
