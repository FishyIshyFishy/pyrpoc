"""``ShownUnit``: one unit as the Autonomy panel draws and lists it."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.plugins.program_io.outputs import Outline, Unit, UnitState, UnitView


@dataclass(frozen=True)
class ShownUnit:
    """A unit on screen: its key, the number it is labelled with, its outline, its group, and
    what a loop is doing with it (None before any loop runs on it)."""

    key: str
    number: int
    outline: Outline
    group: str
    state: UnitState | None
    next_due_s: float | None

    @property
    def unit(self) -> Unit:
        return Unit(self.key, self.outline, self.group)


def drawn(units: tuple[Unit, ...]) -> tuple[ShownUnit, ...]:
    """Units no loop has started on, numbered in order, as a loop starting on them will."""
    return tuple(
        ShownUnit(unit.key, index + 1, unit.outline, unit.group, None, None)
        for index, unit in enumerate(units)
    )


def in_loop(units: tuple[UnitView, ...]) -> tuple[ShownUnit, ...]:
    """A loop's units that have not been removed, where it last tracked each."""
    return tuple(
        ShownUnit(view.key, view.number, view.outline, view.group, view.state, view.next_due_s)
        for view in units
        if view.state is not UnitState.REMOVED
    )
