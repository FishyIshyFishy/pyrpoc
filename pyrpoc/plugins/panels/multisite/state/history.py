"""What the loop's log says looking back: which events a person should see, how each went, a
routine's last scan, and the latest focus check."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import Enum

from pyrpoc.plugins.program_io.outputs import (
    FocusChecked,
    FocusingAtUnit,
    LoopEvent,
    LoopFinished,
    LoopResumed,
    LoopStarted,
    Measured,
    MovingToUnit,
    RoutineEnded,
    RoutineFailed,
    ScanningUnit,
    StepDue,
    UnitRefused,
    UnitScanned,
)

from .routines import RoutineKey


class Tone(Enum):
    """How an event went, with the mark shown beside it so colour is never alone."""

    GOOD = "✓"
    WARNING = "!"
    BAD = "✕"
    CHANGE = "•"


# Logged on every step; the dish and queue already show them as they happen.
ROUTINE = (StepDue, MovingToUnit, FocusingAtUnit, ScanningUnit, Measured)


def tone_of(event: LoopEvent) -> Tone:
    match event:
        case UnitScanned():
            return Tone.GOOD if event.found else Tone.WARNING
        case UnitRefused() | RoutineFailed():
            return Tone.BAD
        case RoutineEnded() | LoopFinished():
            return Tone.GOOD
        case _:
            return Tone.CHANGE


def newest(events: Sequence[LoopEvent]) -> Iterator[LoopEvent]:
    return (events[index] for index in range(len(events) - 1, -1, -1))


def notable(events: Sequence[LoopEvent], count: int) -> list[LoopEvent]:
    """The latest ``count`` events worth a person's eye, newest first."""
    found: list[LoopEvent] = []
    for event in newest(events):
        if not isinstance(event, ROUTINE):
            found.append(event)
            if len(found) == count:
                break
    return found


def unit_of(event: LoopEvent) -> int | None:
    """The number of the unit an event is about, if it is about one."""
    match event:
        case LoopStarted() | LoopResumed() | LoopFinished() | UnitRefused():
            return None
        case _:
            return event.unit


def last_scan(events: Sequence[LoopEvent], key: RoutineKey) -> UnitScanned | None:
    for event in newest(events):
        if isinstance(event, UnitScanned) and (event.unit, event.routine) == key:
            return event
    return None


def last_focus(events: Sequence[LoopEvent]) -> FocusChecked | None:
    for event in newest(events):
        if isinstance(event, FocusChecked):
            return event
    return None


def scans(events: Sequence[LoopEvent]) -> tuple[int, int]:
    """How many scans the loop has taken, and how many of them lost their unit."""
    taken = [event for event in events if isinstance(event, UnitScanned)]
    return len(taken), sum(1 for event in taken if not event.found)
