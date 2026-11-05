"""``Resumption``: what a loop cut short had reached, worked out from its events, for another
run to carry on from."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from pyrpoc.plugins.program_io.outputs import (
    CommandRefused,
    FocusChecked,
    LoopEvent,
    LoopFinished,
    LoopResumed,
    LoopStarted,
    Measured,
    RoutineEnded,
    RoutineFailed,
    RunNowAsked,
    StepDue,
    Unit,
    UnitEdited,
    UnitJoined,
    UnitLeft,
    UnitPaused,
    UnitRecord,
    UnitResumed,
    UnitScanned,
)

from .routine import Recorded, RoutineHistory


@dataclass
class ThreadProgress:
    """How far one routine got on one unit: when it was next due, whether it ended, how many
    frames it took, and its history."""

    due_s: float
    ended: bool = False
    frames: int = 0
    scans_s: list[float] = field(default_factory=list)
    records: list[Recorded] = field(default_factory=list)

    def history(self) -> RoutineHistory:
        return RoutineHistory(tuple(self.scans_s), tuple(self.records))


@dataclass
class Placed:
    """Where a unit was last known to be, and when; None for a unit not yet scanned, which
    starts again from the survey."""

    x_um: float
    y_um: float
    seen_s: float


@dataclass
class Resumption:
    """Every unit by number, each routine's progress by (unit, routine), where each unit was
    last known to be, and the focus offset with when it was last checked."""

    records: dict[int, UnitRecord] = field(default_factory=dict)
    threads: dict[tuple[int, str], ThreadProgress] = field(default_factory=dict)
    placed: dict[int, Placed | None] = field(default_factory=dict)
    focus_offset_um: float = 0.0
    focus_checked_s: float | None = None


def resumption(events: Iterable[LoopEvent]) -> Resumption:
    """What the loop that logged ``events`` had reached."""
    state = Resumption()
    for event in events:
        take(state, event)
    return state


def take(state: Resumption, event: LoopEvent) -> None:
    match event:
        case UnitJoined():
            unit = Unit(event.key, event.outline, event.group)
            state.records[event.unit] = UnitRecord(event.unit, unit, event.time_s, None, False)
            state.placed[event.unit] = None
        case UnitEdited():
            record = state.records[event.unit]
            unit = Unit(record.unit.key, event.outline, event.group)
            state.records[event.unit] = replace(record, unit=unit)
            state.placed[event.unit] = Placed(*unit.centre, event.time_s)
        case UnitLeft():
            state.records[event.unit] = replace(state.records[event.unit], left_s=event.time_s)
        case UnitPaused() | UnitResumed():
            paused = isinstance(event, UnitPaused)
            state.records[event.unit] = replace(state.records[event.unit], paused=paused)
        case StepDue():
            progress = state.threads.setdefault((event.unit, event.routine), ThreadProgress(0.0))
            progress.due_s = event.due_s
        case UnitScanned():
            take_scan(state, event)
        case Measured():
            record = Recorded(event.name, event.time_s, event.values)
            state.threads[(event.unit, event.routine)].records.append(record)
        case RoutineEnded() | RoutineFailed():
            state.threads[(event.unit, event.routine)].ended = True
        case FocusChecked():
            state.focus_offset_um, state.focus_checked_s = event.offset_um, event.time_s
        case LoopStarted() | LoopResumed() | RunNowAsked() | CommandRefused() | LoopFinished():
            pass


def take_scan(state: Resumption, event: UnitScanned) -> None:
    progress = state.threads[(event.unit, event.routine)]
    progress.frames = event.frame + 1
    progress.scans_s.append(event.time_s)
    if event.found:
        state.placed[event.unit] = Placed(event.x_um, event.y_um, event.time_s)
