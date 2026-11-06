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
            return "loop started"
        case LoopResumed():
            return "loop carried on"
        case UnitJoined():
            return f"site {event.unit} added"
        case UnitEdited():
            return f"site {event.unit} changed"
        case UnitLeft():
            return f"site {event.unit} removed"
        case UnitPaused():
            return f"site {event.unit} paused"
        case UnitResumed():
            return f"site {event.unit} resumed"
        case RunNowAsked():
            return f"site {event.unit} to be visited next"
        case CommandRefused():
            return f"could not {event.command} a site: {event.reason}"
        case StepDue():
            return f"site {event.unit} due"
        case UnitScanned():
            found = "found" if event.found else "lost track"
            return f"site {event.unit} scanned, {found}"
        case Measured():
            return f"site {event.unit} {event.name} measured"
        case FocusChecked():
            return f"site {event.unit} focus at {event.z_um:.1f} µm"
        case RoutineEnded():
            return f"site {event.unit} {event.routine} finished"
        case RoutineFailed():
            return f"site {event.unit} {event.routine} failed: {event.message}"
        case LoopFinished():
            return "loop finished"
