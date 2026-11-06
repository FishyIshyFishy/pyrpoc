"""One line of text for each kind of loop event, as the Autonomy panel lists them."""

from __future__ import annotations

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
    UnitEdited,
    UnitJoined,
    UnitLeft,
    UnitPaused,
    UnitResumed,
    UnitScanned,
)


def event_text(event: LoopEvent, started_s: float) -> str:
    """``event`` as a line starting with its time since the loop started."""
    minutes = (event.time_s - started_s) / 60.0
    return f"{minutes:5.1f} min  {what_happened(event)}"


def what_happened(event: LoopEvent) -> str:
    match event:
        case LoopStarted():
            return f"loop started: {', '.join(event.routines)}"
        case LoopResumed():
            return "loop carried on"
        case UnitJoined():
            return f"unit {event.unit} added"
        case UnitEdited():
            return f"unit {event.unit} edited"
        case UnitLeft():
            return f"unit {event.unit} removed"
        case UnitPaused():
            return f"unit {event.unit} paused"
        case UnitResumed():
            return f"unit {event.unit} resumed"
        case RunNowAsked():
            return f"unit {event.unit} asked to run now"
        case CommandRefused():
            return f"{event.command} refused: {event.reason}"
        case StepDue():
            return f"unit {event.unit} {event.routine} due"
        case UnitScanned():
            found = "found" if event.found else "lost"
            return f"unit {event.unit} scanned, {found}"
        case Measured():
            return f"unit {event.unit} {event.name} measured"
        case FocusChecked():
            return f"unit {event.unit} focus at {event.z_um:.1f} µm"
        case RoutineEnded():
            return f"unit {event.unit} {event.routine} done"
        case RoutineFailed():
            return f"unit {event.unit} {event.routine} failed: {event.message}"
        case LoopFinished():
            return "loop finished"
