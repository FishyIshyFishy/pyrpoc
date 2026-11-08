"""``Snapshot``: everything the panel shows at one moment, worked out once and handed to every
part of it."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.plugins.program_io.outputs import LoopActivity, LoopView, UnitView

from .routines import RoutineKey, RoutineStatus
from .run_order import QueueRow, Section, queue_rows


@dataclass(frozen=True)
class Snapshot:
    """The loop as of ``now_s`` on the instrument's clock: what its log says, its open routines,
    and the queue they make in the person's order."""

    view: LoopView
    statuses: tuple[RoutineStatus, ...]
    now_s: float
    order: tuple[RoutineKey, ...]
    rows: tuple[QueueRow, ...]

    @property
    def doing(self) -> LoopActivity | None:
        return self.view.doing

    def unit(self, number: int) -> UnitView | None:
        return next((unit for unit in self.view.units if unit.number == number), None)

    def coming(self, count: int) -> tuple[QueueRow, ...]:
        """The next ``count`` routines to run after the one running now."""
        waiting = [row for row in self.rows if row.section not in (Section.NOW, Section.PAUSED)]
        return tuple(waiting[:count])


def snapshot_of(
    view: LoopView,
    statuses: tuple[RoutineStatus, ...],
    now_s: float,
    order: tuple[RoutineKey, ...],
) -> Snapshot:
    doing = view.doing
    running = None if doing is None else (doing.unit, doing.routine)
    return Snapshot(view, statuses, now_s, order, queue_rows(statuses, running, now_s, order))
