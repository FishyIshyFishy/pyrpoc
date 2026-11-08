"""Units drawn but not yet taken by the loop, which takes them only between steps, so a drawn shape
stays on the dish until the loop joins or refuses it."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.plugins.program_io.outputs import LoopView, Unit, UnitRefused

# How long a drawn unit waits for the loop before the panel stops showing it.
GIVE_UP_S = 30.0


@dataclass(frozen=True)
class Draft:
    unit: Unit
    sent_s: float


def answered(draft: Draft, view: LoopView) -> bool:
    """Whether the loop has joined or refused ``draft``'s unit."""
    key = draft.unit.key
    if any(unit.key == key for unit in view.units):
        return True
    # A refusal is only in the log, which the view keeps the latest of.
    return any(isinstance(event, UnitRefused) and event.key == key for event in view.recent)


class Drafts:
    def __init__(self) -> None:
        self.waiting: list[Draft] = []

    def add(self, unit: Unit, now_s: float) -> None:
        self.waiting.append(Draft(unit, now_s))

    def settle(self, view: LoopView, now_s: float) -> None:
        self.waiting = [
            draft
            for draft in self.waiting
            if now_s - draft.sent_s <= GIVE_UP_S and not answered(draft, view)
        ]
