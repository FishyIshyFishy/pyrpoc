"""The menu a right-click on a unit opens, wherever the unit is shown: what a person can ask of
the loop for it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMenu, QWidget

from .state.pending import Change
from .state.standing import Standing, UnitRow

# Units the loop still runs routines on, which can be paused or run next.
OPEN = (Standing.RUNNING, Standing.WAITING, Standing.LOST, Standing.PAUSED)


class UnitActions(Protocol):
    def run_next(self, number: int) -> None: ...

    def pause(self, number: int, paused: bool) -> None: ...

    def remove(self, number: int) -> None: ...

    def drop_drawn(self, key: str) -> None: ...

    def locate(self, key: str) -> None: ...


def action(menu: QMenu, text: str, enabled: bool, does: Callable[[], None]) -> None:
    # Connected rather than read from exec()'s return: Qt hides the menu before it emits, so a
    # dialog opened from it opens with the menu already gone.
    item = QAction(text, menu)
    item.setEnabled(enabled)
    item.triggered.connect(lambda _checked=False: does())
    menu.addAction(item)


def unit_menu(
    parent: QWidget,
    row: UnitRow,
    running: bool,
    pending: Change | None,
    actions: UnitActions,
    from_dish: bool,
) -> QMenu:
    """The menu for ``row``'s unit; changes wait while the loop has not taken the last one."""
    menu = QMenu(parent)
    group = f" ({row.group})" if row.group else ""
    # A title line, since not every style draws a menu's section titles.
    action(menu, f"{row.name}{group}", False, lambda: None)
    menu.addSeparator()
    number = row.number
    if number is None:
        action(menu, "Remove drawn unit", True, lambda: actions.drop_drawn(row.key))
    else:
        free = running and pending is None and row.standing in OPEN
        queued = row.next is not None and (row.next.place == 0 or row.next.urgent)
        paused = row.standing is Standing.PAUSED
        action(
            menu, "Run next", free and not paused and not queued, lambda: actions.run_next(number)
        )
        action(
            menu, "Resume" if paused else "Pause", free, lambda: actions.pause(number, not paused)
        )
        action(menu, "Remove from loop...", free, lambda: actions.remove(number))
    if not from_dish:
        menu.addSeparator()
        action(menu, "Show on dish", True, lambda: actions.locate(row.key))
    return menu
