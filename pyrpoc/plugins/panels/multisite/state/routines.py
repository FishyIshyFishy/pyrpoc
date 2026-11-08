"""``RoutineStatus``: one open routine on one unit, as the loop's log tells it."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.plugins.program_io.outputs.loop_view import LoopFold

# A routine on a unit: the unit's number and the routine's name.
RoutineKey = tuple[int, str]


@dataclass(frozen=True)
class RoutineStatus:
    """A routine that has not ended on one unit: when it is next due on the instrument's clock,
    whether its unit is paused or was lost at its last scan, and whether it was asked to run
    now."""

    unit: int
    routine: str
    due_s: float
    paused: bool
    lost: bool
    urgent: bool

    @property
    def key(self) -> RoutineKey:
        return (self.unit, self.routine)


def routine_statuses(fold: LoopFold) -> tuple[RoutineStatus, ...]:
    """Every open routine of every unit still in the loop, paused ones included."""
    return tuple(
        RoutineStatus(number, routine, due_s, unit.paused, unit.lost, routine in unit.urgent)
        for number, unit in fold.units.items()
        if not unit.left
        for routine, due_s in unit.due.items()
    )
