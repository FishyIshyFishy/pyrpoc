"""Changes sent to the loop that it has not taken up yet, since it takes them only between
steps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from pyrpoc.plugins.program_io.outputs import LoopView, UnitState

from .routines import RoutineStatus

# How long a change is shown as on its way before the panel stops waiting for it.
GIVE_UP_S = 30.0


class Change(Enum):
    PAUSE = "pausing"
    RESUME = "resuming"
    RUN_NOW = "running next"
    REMOVE = "removing"


@dataclass(frozen=True)
class Sent:
    change: Change
    sent_s: float


def taken_up(change: Change, state: UnitState | None, own: Sequence[RoutineStatus]) -> bool:
    """Whether the loop shows ``change`` done, from its unit's state and open routines."""
    if state is None or state in (UnitState.REMOVED, UnitState.DONE):
        return True
    match change:
        case Change.PAUSE:
            return state is UnitState.PAUSED
        case Change.RESUME:
            return state is not UnitState.PAUSED
        case Change.RUN_NOW:
            return any(status.urgent for status in own)
        case Change.REMOVE:
            return False


class Pending:
    """The latest change sent for each unit, until the loop shows it taken up."""

    def __init__(self) -> None:
        self.sent: dict[int, Sent] = {}

    def send(self, unit: int, change: Change, now_s: float) -> None:
        self.sent[unit] = Sent(change, now_s)

    def of(self, unit: int) -> Change | None:
        sent = self.sent.get(unit)
        return None if sent is None else sent.change

    def settle(self, view: LoopView, statuses: tuple[RoutineStatus, ...], now_s: float) -> None:
        """Forget each change the loop has taken up, or that has waited too long."""
        states = {unit.number: unit.state for unit in view.units}
        for unit, sent in list(self.sent.items()):
            own = [status for status in statuses if status.unit == unit]
            if now_s - sent.sent_s > GIVE_UP_S or taken_up(sent.change, states.get(unit), own):
                del self.sent[unit]
