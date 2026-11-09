"""What the panel shows of a loop at one moment: its queue, in the order the loop runs it, and
every unit with what it does next."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from pyrpoc.plugins.program_io.outputs import (
    LoopActivity,
    LoopView,
    MovingToUnit,
    Outline,
    ScanningUnit,
    Unit,
    UnitState,
    UnitView,
)

from .outline import middle


class Standing(Enum):
    """Where a unit stands with the loop."""

    DRAWN = "Drawn"
    REFUSED = "Refused"
    RUNNING = "Running"
    WAITING = "Waiting"
    LOST = "Lost"
    PAUSED = "Paused"
    DONE = "Done"


@dataclass(frozen=True)
class Snapshot:
    """A loop as of ``now_s`` on the instrument's clock: what its log says, or None before a
    loop has run over the survey; whether its run is still going; the units given to it that it
    has not joined; and why the loop refused any of them, by key."""

    view: LoopView | None
    now_s: float
    running: bool
    drawn: tuple[Unit, ...]
    refused: Mapping[str, str]

    @property
    def doing(self) -> LoopActivity | None:
        """What the loop is doing now; a run cut short is doing nothing, whatever it last
        logged."""
        if self.view is None or not self.running:
            return None
        return self.view.doing


@dataclass(frozen=True)
class QueueRow:
    """One routine on one unit in the queue: ``place`` 0 is running now, then 1 runs next.
    ``time_s``, on the instrument's clock, is when it falls due, or for the one running, when
    its current phase began."""

    place: int
    unit: int
    key: str
    routine: str
    time_s: float
    urgent: bool


@dataclass(frozen=True)
class UnitRow:
    """One unit: its key, the number the loop gave it (None until it joins), its name, its
    group, its outline where it was last tracked to, where it stands, and its first row in the
    queue."""

    key: str
    number: int | None
    name: str
    group: str
    outline: Outline
    standing: Standing
    next: QueueRow | None


def queue_rows(snapshot: Snapshot) -> tuple[QueueRow, ...]:
    """The routine running now, then every routine waiting, in the order the loop takes them:
    those asked to run now first, then the earliest due. A paused unit's are not queued."""
    view = snapshot.view
    if view is None:
        return ()
    keys = {unit.number: unit.key for unit in view.units}
    doing = snapshot.doing
    rows: list[QueueRow] = []
    if doing is not None:
        rows.append(QueueRow(0, doing.unit, keys[doing.unit], doing.routine, doing.time_s, False))
    for entry in view.queue:
        if doing is not None and (entry.unit, entry.routine) == (doing.unit, doing.routine):
            continue
        place = len(rows) + (0 if doing is not None else 1)
        due_s = view.started_s + entry.due_s
        key = keys[entry.unit]
        rows.append(QueueRow(place, entry.unit, key, entry.routine, due_s, entry.urgent))
    return tuple(rows)


def standing_of(unit: UnitView, doing: LoopActivity | None) -> Standing:
    if doing is not None and doing.unit == unit.number:
        return Standing.RUNNING
    match unit.state:
        case UnitState.WAITING:
            return Standing.WAITING
        case UnitState.LOST:
            return Standing.LOST
        case UnitState.PAUSED:
            return Standing.PAUSED
        case _:
            return Standing.DONE


def unit_rows(snapshot: Snapshot, queue: tuple[QueueRow, ...]) -> tuple[UnitRow, ...]:
    """Every unit in the loop, then those drawn and not joined; units removed are left out."""
    rows: list[UnitRow] = []
    view = snapshot.view
    if view is not None:
        for unit in view.units:
            if unit.state is UnitState.REMOVED:
                continue
            first = next((row for row in queue if row.unit == unit.number), None)
            standing = standing_of(unit, snapshot.doing)
            name = f"Unit {unit.number}"
            rows.append(
                UnitRow(unit.key, unit.number, name, unit.group, unit.outline, standing, first)
            )
    # A drawn unit has no number until the loop gives it one, so it is named by its place.
    for index, unit in enumerate(snapshot.drawn, start=1):
        standing = Standing.REFUSED if unit.key in snapshot.refused else Standing.DRAWN
        name = f"New {index}"
        rows.append(UnitRow(unit.key, None, name, unit.group, unit.outline, standing, None))
    return tuple(rows)


def working_at(snapshot: Snapshot) -> tuple[tuple[float, float], bool] | None:
    """Where the loop is working in dish µm, and whether it is scanning there, or None between
    steps."""
    doing, view = snapshot.doing, snapshot.view
    if doing is None or view is None:
        return None
    scanning = isinstance(doing, ScanningUnit)
    if isinstance(doing, MovingToUnit):
        return (doing.x_um, doing.y_um), scanning
    unit = next(unit for unit in view.units if unit.number == doing.unit)
    return middle(unit.outline), scanning
