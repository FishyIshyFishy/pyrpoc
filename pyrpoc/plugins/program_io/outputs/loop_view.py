"""``LoopView``: what a loop's events say it is doing now, and ``LoopFold``, which works it out
from them."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum

from .loop_events import (
    FocusChecked,
    FocusingAtUnit,
    LoopActivity,
    LoopEvent,
    LoopFinished,
    LoopResumed,
    LoopStarted,
    Measured,
    MovingToUnit,
    RoutineEnded,
    RoutineFailed,
    RunNowAsked,
    ScanningUnit,
    StepDue,
    UnitEdited,
    UnitJoined,
    UnitLeft,
    UnitPaused,
    UnitRefused,
    UnitResumed,
    UnitScanned,
)
from .units import Outline, Unit

# How many of the latest events a view carries.
RECENT = 12


class UnitState(Enum):
    WAITING = "waiting"
    PAUSED = "paused"
    LOST = "lost"
    DONE = "done"
    REMOVED = "removed"


@dataclass(frozen=True)
class Series:
    """One measurement a routine recorded of a unit: each time, in seconds since the loop
    started, with one value per channel."""

    routine: str
    name: str
    points: tuple[tuple[float, tuple[float, ...]], ...]


@dataclass(frozen=True)
class UnitView:
    """One unit as the log tells it: its number and key, its group, its outline where it was
    last tracked to, its state, when it is next due in seconds since the loop started, and
    every measurement of it."""

    number: int
    key: str
    group: str
    outline: Outline
    state: UnitState
    next_due_s: float | None
    series: tuple[Series, ...]


@dataclass(frozen=True)
class QueueEntry:
    """A routine step that is due: whose, in which routine, and when, in seconds since the loop
    started."""

    unit: int
    routine: str
    due_s: float


@dataclass(frozen=True)
class LoopView:
    """A loop as of its latest event: when it started on the instrument's clock and how long it
    has run, its channels and routines, every unit, the queue with those asked to run now
    first and then the soonest, what it is doing now (None between scans), and its latest
    events, newest first."""

    started_s: float
    elapsed_s: float
    channels: tuple[str, ...]
    routines: tuple[str, ...]
    units: tuple[UnitView, ...]
    queue: tuple[QueueEntry, ...]
    doing: LoopActivity | None
    recent: tuple[LoopEvent, ...]


@dataclass
class UnitFold:
    """One unit while folding: as last drawn, where it was last tracked, and its routines."""

    unit: Unit
    tracked: tuple[float, float]
    lost: bool = False
    paused: bool = False
    left: bool = False
    # Each open routine's next due time, and the routines asked to run now.
    due: dict[str, float] = field(default_factory=dict)
    urgent: set[str] = field(default_factory=set)
    series: dict[tuple[str, str], list[tuple[float, tuple[float, ...]]]] = field(
        default_factory=dict
    )

    def state(self) -> UnitState:
        if self.left:
            return UnitState.REMOVED
        if not self.due:
            return UnitState.DONE
        if self.paused:
            return UnitState.PAUSED
        return UnitState.LOST if self.lost else UnitState.WAITING


class LoopFold:
    """The log's events taken in order into what the loop is doing."""

    def __init__(self) -> None:
        self.started_s = 0.0
        self.latest_s = 0.0
        self.channels: tuple[str, ...] = ()
        self.routines: tuple[str, ...] = ()
        self.units: dict[int, UnitFold] = {}
        self.doing: LoopActivity | None = None
        self.recent: deque[LoopEvent] = deque(maxlen=RECENT)

    def take(self, event: LoopEvent) -> None:
        self.latest_s = event.time_s
        self.recent.append(event)
        match event:
            case LoopStarted():
                self.started_s, self.channels = event.time_s, event.channels
                self.routines = event.routines
            case UnitJoined():
                unit = Unit(event.key, event.outline, event.group, False, 0)
                self.units[event.unit] = UnitFold(unit, unit.centre)
            case UnitEdited():
                fold = self.units[event.unit]
                fold.unit = replace(fold.unit, outline=event.outline, group=event.group)
                fold.tracked, fold.lost = fold.unit.centre, False
            case UnitLeft():
                self.units[event.unit].left = True
                self.units[event.unit].due.clear()
            case UnitPaused() | UnitResumed():
                self.units[event.unit].paused = isinstance(event, UnitPaused)
            case RunNowAsked():
                self.units[event.unit].urgent |= set(self.units[event.unit].due)
            case StepDue():
                self.units[event.unit].due[event.routine] = event.due_s
            case MovingToUnit() | FocusingAtUnit() | ScanningUnit():
                self.doing = event
            case UnitScanned():
                self.doing = None
                self.take_scan(self.units[event.unit], event)
            case Measured():
                points = self.units[event.unit].series.setdefault((event.routine, event.name), [])
                points.append((event.time_s - self.started_s, event.values))
            case RoutineEnded() | RoutineFailed():
                self.units[event.unit].due.pop(event.routine, None)
            case LoopResumed() | LoopFinished():
                # A loop carried on, or ended, is between scans whatever it was doing.
                self.doing = None
            case FocusChecked() | UnitRefused():
                pass

    def take_scan(self, fold: UnitFold, event: UnitScanned) -> None:
        fold.tracked, fold.lost = (event.x_um, event.y_um), not event.found
        fold.urgent.discard(event.routine)

    def view(self) -> LoopView:
        return LoopView(
            self.started_s,
            self.latest_s - self.started_s,
            self.channels,
            self.routines,
            tuple(self.unit_view(number, fold) for number, fold in self.units.items()),
            self.queue(),
            self.doing,
            tuple(reversed(self.recent)),
        )

    def unit_view(self, number: int, fold: UnitFold) -> UnitView:
        due = min(fold.due.values(), default=None)
        series = tuple(
            Series(routine, name, tuple(points)) for (routine, name), points in fold.series.items()
        )
        return UnitView(
            number,
            fold.unit.key,
            fold.unit.group,
            fold.unit.moved_to(*fold.tracked).outline,
            fold.state(),
            None if due is None else due - self.started_s,
            series,
        )

    def queue(self) -> tuple[QueueEntry, ...]:
        """The open, unpaused routines: those asked to run now first, then the earliest due."""
        entries = [
            (routine not in fold.urgent, due, QueueEntry(number, routine, due - self.started_s))
            for number, fold in self.units.items()
            if not fold.paused
            for routine, due in fold.due.items()
        ]
        return tuple(entry for _urgent, _due, entry in sorted(entries, key=lambda e: e[:2]))
