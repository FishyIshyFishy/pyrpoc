"""Changes sent to a loop that it has not taken yet: it takes them only between its steps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from pyrpoc.plugins.program_io.outputs import (
    LoopEvent,
    RunNowAsked,
    UnitLeft,
    UnitPaused,
    UnitResumed,
)

# How long a change is shown as on its way before the panel stops waiting for it.
GIVE_UP_S = 30.0


class Change(Enum):
    PAUSE = "Pausing…"
    RESUME = "Resuming…"
    RUN_NEXT = "Moving to the front…"
    REMOVE = "Removing…"


@dataclass(frozen=True)
class Sent:
    change: Change
    wall_s: float


# The event a loop logs when it takes each change.
TAKEN_BY: dict[Change, type[LoopEvent]] = {
    Change.PAUSE: UnitPaused,
    Change.RESUME: UnitResumed,
    Change.RUN_NEXT: RunNowAsked,
    Change.REMOVE: UnitLeft,
}


class Pending:
    """The latest change sent for each unit, by its number, until the loop logs it taken."""

    def __init__(self) -> None:
        self.sent: dict[int, Sent] = {}

    def send(self, unit: int, change: Change, wall_s: float) -> None:
        self.sent[unit] = Sent(change, wall_s)

    def of(self, unit: int) -> Change | None:
        sent = self.sent.get(unit)
        return None if sent is None else sent.change

    def settle(self, events: Sequence[LoopEvent], wall_s: float) -> None:
        """Forget each change the new ``events`` show taken, and each that waited too long."""
        for event in events:
            if not isinstance(event, (UnitPaused, UnitResumed, RunNowAsked, UnitLeft)):
                continue
            sent = self.sent.get(event.unit)
            # A unit that left has taken every change it will.
            if sent is not None and isinstance(event, (TAKEN_BY[sent.change], UnitLeft)):
                del self.sent[event.unit]
        for unit, sent in list(self.sent.items()):
            if wall_s - sent.wall_s > GIVE_UP_S:
                del self.sent[unit]
