"""``Activity``: the things a loop did that a person needs to know, as lines of text."""

from __future__ import annotations

from collections import deque

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

# Lines kept, newest first: enough to scroll back over an afternoon of changes.
KEPT = 200


class Activity:
    """Takes a loop's events in order and keeps a line for each that matters: sites added,
    changed, removed, paused and resumed, a site's tracking lost or found again, routines
    finishing or failing, refused changes, and the loop starting and ending. Routine visits,
    measurements and focus checks are left out: they happen for every site on every visit."""

    def __init__(self) -> None:
        self.taken = 0
        self.started_s = 0.0
        self.lines: deque[str] = deque(maxlen=KEPT)
        # Whether each site was found on its last scan, so only a change is reported.
        self.found: dict[int, bool] = {}

    def take(self, events: list[LoopEvent]) -> None:
        for event in events:
            if isinstance(event, LoopStarted):
                self.started_s = event.time_s
            text = self.what_happened(event)
            if text is not None:
                minutes = (event.time_s - self.started_s) / 60.0
                self.lines.appendleft(f"{minutes:5.1f} min   {text}")
        self.taken += len(events)

    def what_happened(self, event: LoopEvent) -> str | None:
        match event:
            case LoopStarted():
                return "loop started"
            case LoopResumed():
                return "loop carried on from where it stopped"
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
            case UnitScanned():
                return self.tracking(event)
            case RoutineEnded():
                return f"site {event.unit} finished {event.routine}"
            case RoutineFailed():
                return f"site {event.unit} {event.routine} failed: {event.message}"
            case LoopFinished():
                return "loop finished"
            case StepDue() | Measured() | FocusChecked():
                return None

    def tracking(self, event: UnitScanned) -> str | None:
        """A line when a site is lost or found again, none while it stays as it was."""
        was = self.found.get(event.unit, True)
        self.found[event.unit] = event.found
        if event.found == was:
            return None
        return f"site {event.unit} found again" if event.found else f"site {event.unit} lost track"
