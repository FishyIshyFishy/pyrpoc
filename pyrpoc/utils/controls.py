"""The common kinds of ``Control`` a runner offers: ``Button``, pressed, and ``Toggle``, on or
off."""

from __future__ import annotations

from collections.abc import Callable

from pyrpoc.core.runner import Control


class Button(Control):
    """Pressed, not held. ``icon`` names a file in ``utils/assets/`` without its extension."""

    def __init__(self, label: str, on_press: Callable[[], None], *, icon: str | None, tooltip: str):
        self.label = label
        self.icon = icon
        self.tooltip = tooltip
        self.on_press = on_press

    def press(self) -> None:
        self.on_press()


class Toggle(Control):
    """On or off. ``flip`` is the user moving it and calls ``on_change``; ``set`` is the runner
    moving it and calls nothing. ``icon`` names a file in ``utils/assets/``."""

    def __init__(
        self, label: str, on_change: Callable[[bool], None], *, icon: str | None, tooltip: str
    ):
        self.label = label
        self.icon = icon
        self.tooltip = tooltip
        self.on_change = on_change
        self.checked = False
        self._watchers: list[Callable[[], None]] = []

    def set(self, checked: bool) -> None:
        # Calls nothing, so a runner turning its own toggle off never re-enters its handler.
        if checked != self.checked:
            self.checked = checked
            for callback in list(self._watchers):
                callback()

    def flip(self, checked: bool) -> None:
        self.set(checked)
        self.on_change(self.checked)

    def watch(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` whenever it is moved."""
        self._watchers.append(callback)

    def unwatch(self, callback: Callable[[], None]) -> None:
        self._watchers.remove(callback)
