"""``Feedback``: what to say while a change sent to a running loop waits to be taken, and once it
is taken or refused."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.plugins.program_io.outputs import (
    CommandRefused,
    LoopEvent,
    RunNowAsked,
    UnitEdited,
    UnitJoined,
    UnitLeft,
    UnitPaused,
    UnitResumed,
)
from pyrpoc.plugins.program_io.requests import (
    AddUnit,
    EditUnit,
    PauseUnit,
    RemoveUnit,
    ResumeUnit,
    RunUnitNow,
    UnitCommand,
    command_key,
)

# The verb each command is called by on screen, as in "could not pause site 2".
VERBS: dict[type, str] = {
    AddUnit: "add",
    EditUnit: "change",
    RemoveUnit: "remove",
    PauseUnit: "pause",
    ResumeUnit: "resume",
    RunUnitNow: "visit",
}
# While waiting, and the event that says each command was carried out.
ASKED: dict[type, str] = {
    AddUnit: "Adding the new site to the loop…",
    EditUnit: "Updating site {n}…",
    RemoveUnit: "Removing site {n}…",
    PauseUnit: "Pausing site {n}…",
    ResumeUnit: "Resuming site {n}…",
    RunUnitNow: "Asking the loop to visit site {n} next…",
}
DONE: dict[type, tuple[type, str]] = {
    AddUnit: (UnitJoined, "Site {n} added to the loop."),
    EditUnit: (UnitEdited, "Site {n} updated."),
    RemoveUnit: (UnitLeft, "Site {n} removed. Its images so far are kept."),
    PauseUnit: (UnitPaused, "Site {n} paused."),
    ResumeUnit: (UnitResumed, "Site {n} resumed."),
    RunUnitNow: (RunNowAsked, "Site {n} will be visited next."),
}


@dataclass(frozen=True)
class Waiting:
    """A command sent and not yet taken: its kind, its site's key, and its number, which a new
    site does not have until it joins."""

    kind: type
    key: str
    number: int | None


class Feedback:
    """Remembers the last command sent, and says what became of it."""

    def __init__(self) -> None:
        self.waiting: Waiting | None = None

    def ask(self, command: UnitCommand, number: int | None) -> str:
        """Remember ``command``, sent for site ``number``, and say it is on its way."""
        self.waiting = Waiting(type(command), command_key(command), number)
        return ASKED[type(command)].format(n=number)

    def take(self, events: list[LoopEvent]) -> str | None:
        """What became of the waiting command, if one of ``events`` tells."""
        waiting = self.waiting
        if waiting is None:
            return None
        done_kind, done_text = DONE[waiting.kind]
        for event in events:
            if isinstance(event, CommandRefused) and event.key == waiting.key:
                self.waiting = None
                site = "the new site" if waiting.number is None else f"site {waiting.number}"
                return f"The loop could not {VERBS[waiting.kind]} {site}: {event.reason}."
            site = site_of(event, waiting)
            if isinstance(event, done_kind) and site is not None:
                self.waiting = None
                return done_text.format(n=site)
        return None


def site_of(event: LoopEvent, waiting: Waiting) -> int | None:
    """The number of the waiting command's site, if ``event`` is about it."""
    match event:
        case UnitJoined() if event.key == waiting.key:
            return event.unit
        case UnitEdited() | UnitLeft() | UnitPaused() | UnitResumed() | RunNowAsked():
            return event.unit if event.unit == waiting.number else None
        case _:
            return None
