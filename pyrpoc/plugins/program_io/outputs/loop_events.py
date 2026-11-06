"""The events a ``LoopLogOutput`` holds, one class for each kind. Units are named by the number
the loop gave them; times are on the instrument's clock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .units import Outline


@dataclass(frozen=True)
class LoopStarted:
    kind: ClassVar[str] = "started"
    time_s: float
    channels: tuple[str, ...]
    routines: tuple[str, ...]

    def text(self) -> str:
        return f"started, running {', '.join(self.routines)}"


@dataclass(frozen=True)
class LoopResumed:
    """The loop carried on after it was cut short, from everything logged before."""

    kind: ClassVar[str] = "resumed"
    time_s: float

    def text(self) -> str:
        return "carried on the loop cut short in this folder"


@dataclass(frozen=True)
class UnitJoined:
    """A unit was given its number and its routines started, from this outline."""

    kind: ClassVar[str] = "joined"
    time_s: float
    unit: int
    key: str
    group: str
    outline: Outline

    def text(self) -> str:
        group = f" in {self.group}" if self.group else ""
        return f"unit {self.unit} joined{group}"


@dataclass(frozen=True)
class UnitEdited:
    """A unit took a new group, or a new outline saying where it is now."""

    kind: ClassVar[str] = "edited"
    time_s: float
    unit: int
    group: str
    outline: Outline

    def text(self) -> str:
        return f"unit {self.unit} redrawn"


@dataclass(frozen=True)
class UnitLeft:
    kind: ClassVar[str] = "left"
    time_s: float
    unit: int

    def text(self) -> str:
        return f"unit {self.unit} removed"


@dataclass(frozen=True)
class UnitPaused:
    kind: ClassVar[str] = "paused"
    time_s: float
    unit: int

    def text(self) -> str:
        return f"unit {self.unit} paused"


@dataclass(frozen=True)
class UnitResumed:
    kind: ClassVar[str] = "unpaused"
    time_s: float
    unit: int

    def text(self) -> str:
        return f"unit {self.unit} resumed"


@dataclass(frozen=True)
class RunNowAsked:
    """A unit was asked to run at once, for the ``asked``-th time."""

    kind: ClassVar[str] = "run now"
    time_s: float
    unit: int
    asked: int

    def text(self) -> str:
        return f"unit {self.unit} asked to run now"


@dataclass(frozen=True)
class UnitRefused:
    """A change to the units the loop could not make, to the unit with ``key``, and why."""

    kind: ClassVar[str] = "refused"
    time_s: float
    key: str
    reason: str

    def text(self) -> str:
        return f"a change to a unit was refused: {self.reason}"


@dataclass(frozen=True)
class StepDue:
    kind: ClassVar[str] = "due"
    time_s: float
    unit: int
    routine: str
    due_s: float

    def text(self) -> str:
        return f"unit {self.unit} {self.routine}: next due in {self.due_s - self.time_s:.0f} s"


@dataclass(frozen=True)
class MovingToUnit:
    """The stage began moving to a unit, for a scan by its routine."""

    kind: ClassVar[str] = "moving"
    time_s: float
    unit: int
    routine: str
    x_um: float
    y_um: float

    def text(self) -> str:
        return f"unit {self.unit} {self.routine}: moving to it"


@dataclass(frozen=True)
class FocusingAtUnit:
    """A focus check began at a unit, before its routine's scan."""

    kind: ClassVar[str] = "focusing"
    time_s: float
    unit: int
    routine: str

    def text(self) -> str:
        return f"unit {self.unit} {self.routine}: checking focus"


@dataclass(frozen=True)
class ScanningUnit:
    """A unit's scan for its routine began."""

    kind: ClassVar[str] = "scanning"
    time_s: float
    unit: int
    routine: str

    def text(self) -> str:
        return f"unit {self.unit} {self.routine}: scanning"


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

    def text(self) -> str:
        found = "" if self.found else ", not found where it was"
        return f"unit {self.unit} {self.routine}: frame {self.frame + 1} taken{found}"


@dataclass(frozen=True)
class Measured:
    """A routine recorded a measurement of a unit, one value per channel."""

    kind: ClassVar[str] = "measured"
    time_s: float
    unit: int
    routine: str
    name: str
    values: tuple[float, ...]

    def text(self) -> str:
        return f"unit {self.unit} {self.routine}: {self.name} measured"


@dataclass(frozen=True)
class FocusChecked:
    kind: ClassVar[str] = "focus"
    time_s: float
    unit: int
    routine: str
    z_um: float
    offset_um: float

    def text(self) -> str:
        return f"focus found at z {self.z_um:.1f} µm, {self.offset_um:+.1f} µm off the survey"


@dataclass(frozen=True)
class RoutineEnded:
    kind: ClassVar[str] = "ended"
    time_s: float
    unit: int
    routine: str

    def text(self) -> str:
        return f"unit {self.unit} {self.routine}: done"


@dataclass(frozen=True)
class RoutineFailed:
    kind: ClassVar[str] = "failed"
    time_s: float
    unit: int
    routine: str
    message: str

    def text(self) -> str:
        return f"unit {self.unit} {self.routine}: failed, {self.message}"


@dataclass(frozen=True)
class LoopFinished:
    kind: ClassVar[str] = "finished"
    time_s: float

    def text(self) -> str:
        return "every routine has ended"


# What the loop is doing now: the latest of these, until the unit's scan is logged.
LoopActivity = MovingToUnit | FocusingAtUnit | ScanningUnit
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
    | LoopActivity
    | UnitScanned
    | Measured
    | FocusChecked
    | RoutineEnded
    | RoutineFailed
    | LoopFinished
)
LOOP_EVENTS: dict[str, type[LoopEvent]] = {
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
        MovingToUnit,
        FocusingAtUnit,
        ScanningUnit,
        UnitScanned,
        Measured,
        FocusChecked,
        RoutineEnded,
        RoutineFailed,
        LoopFinished,
    )
}
