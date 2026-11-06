"""The events a ``LoopLogOutput`` holds, one class for each kind, and how each is written as a
line of JSON. Units are named by the number the loop gave them; times are on the instrument's
clock."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from .units import Outline


@dataclass(frozen=True)
class LoopStarted:
    kind: ClassVar[str] = "started"
    time_s: float
    channels: tuple[str, ...]
    routines: tuple[str, ...]


@dataclass(frozen=True)
class LoopResumed:
    """The loop carried on after it was cut short, from everything logged before."""

    kind: ClassVar[str] = "resumed"
    time_s: float


@dataclass(frozen=True)
class UnitJoined:
    """A unit was given its number and its routines started, from this outline."""

    kind: ClassVar[str] = "joined"
    time_s: float
    unit: int
    key: str
    group: str
    outline: Outline


@dataclass(frozen=True)
class UnitEdited:
    """A unit took a new group, or a new outline saying where it is now."""

    kind: ClassVar[str] = "edited"
    time_s: float
    unit: int
    group: str
    outline: Outline


@dataclass(frozen=True)
class UnitLeft:
    kind: ClassVar[str] = "left"
    time_s: float
    unit: int


@dataclass(frozen=True)
class UnitPaused:
    kind: ClassVar[str] = "paused"
    time_s: float
    unit: int


@dataclass(frozen=True)
class UnitResumed:
    kind: ClassVar[str] = "unpaused"
    time_s: float
    unit: int


@dataclass(frozen=True)
class RunNowAsked:
    """A unit was asked to run at once, for the ``asked``-th time."""

    kind: ClassVar[str] = "run now"
    time_s: float
    unit: int
    asked: int


@dataclass(frozen=True)
class UnitRefused:
    """A change to the units the loop could not make, to the unit with ``key``, and why."""

    kind: ClassVar[str] = "refused"
    time_s: float
    key: str
    reason: str


@dataclass(frozen=True)
class StepDue:
    kind: ClassVar[str] = "due"
    time_s: float
    unit: int
    routine: str
    due_s: float


@dataclass(frozen=True)
class UnitScanned:
    """A unit was scanned for a routine: the frame's index among that routine's frames of the
    unit, how late the scan ran, where the unit is now and at what z, how far it moved, whether
    it was found within ``radius_um`` of where it was, and the lines driven."""

    kind: ClassVar[str] = "scanned"
    time_s: float
    unit: int
    routine: str
    frame: int
    late_s: float
    x_um: float
    y_um: float
    z_um: float
    shift_um: tuple[float, float]
    found: bool
    radius_um: float
    lines: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Measured:
    """A routine recorded a measurement of a unit, one value per channel."""

    kind: ClassVar[str] = "measured"
    time_s: float
    unit: int
    routine: str
    name: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class FocusChecked:
    kind: ClassVar[str] = "focus"
    time_s: float
    unit: int
    routine: str
    z_um: float
    offset_um: float


@dataclass(frozen=True)
class RoutineEnded:
    kind: ClassVar[str] = "ended"
    time_s: float
    unit: int
    routine: str


@dataclass(frozen=True)
class RoutineFailed:
    kind: ClassVar[str] = "failed"
    time_s: float
    unit: int
    routine: str
    message: str


@dataclass(frozen=True)
class LoopFinished:
    kind: ClassVar[str] = "finished"
    time_s: float


LoopEvent = (
    LoopStarted
    | LoopResumed
    | UnitJoined
    | UnitEdited
    | UnitLeft
    | UnitPaused
    | UnitResumed
    | RunNowAsked
    | UnitRefused
    | StepDue
    | UnitScanned
    | Measured
    | FocusChecked
    | RoutineEnded
    | RoutineFailed
    | LoopFinished
)
EVENTS: dict[str, type[LoopEvent]] = {
    cls.kind: cls
    for cls in (
        LoopStarted,
        LoopResumed,
        UnitJoined,
        UnitEdited,
        UnitLeft,
        UnitPaused,
        UnitResumed,
        RunNowAsked,
        UnitRefused,
        StepDue,
        UnitScanned,
        Measured,
        FocusChecked,
        RoutineEnded,
        RoutineFailed,
        LoopFinished,
    )
}


def event_to_dict(event: LoopEvent) -> dict[str, Any]:
    return {"event": event.kind, **asdict(event)}


def event_from_dict(raw: dict[str, Any]) -> LoopEvent:
    """An event from its JSON, its lists back as tuples."""
    fields = {key: as_tuples(value) for key, value in raw.items() if key != "event"}
    return EVENTS[raw["event"]](**fields)


def as_tuples(value: Any) -> Any:
    return tuple(as_tuples(each) for each in value) if isinstance(value, list) else value
