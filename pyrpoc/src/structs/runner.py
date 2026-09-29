"""Runners: the ways a program can be started, declared by the program.

A runner is a frozen declaration whose ``attach`` builds per-program state
against a ``RunnerContext``, so one declaration can be a shared class
attribute. The context is the host's side: starting a run, parameters,
controls, and requests for a ``Pick`` that the host routes to whoever can
answer.
Controls are Qt-free descriptors the host's view renders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from .params import BlockMap
from .picks import Pick


class Control:
    """A widget a runner asks for. The view renders it and watches it.

    ``icon`` names a file in ``pyrpoc/assets/`` without its extension. The view
    also disables every control while a run is in progress or something is
    missing, whatever ``enabled`` holds.
    """

    # Whether the view draws it as a button that stays down.
    checkable: ClassVar[bool] = False

    def __init__(self, label: str, *, icon: str | None, tooltip: str):
        self.label = label
        self.icon = icon
        self.tooltip = tooltip
        self.checked = False
        self._enabled = True
        self._watchers: list[Callable[[], None]] = []

    def activate(self, checked: bool) -> None:
        """What the view calls when the user clicks it."""
        raise NotImplementedError

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if value != self._enabled:
            self._enabled = value
            self.changed()

    def watch(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` whenever this control's state changes."""
        self._watchers.append(callback)

    def unwatch(self, callback: Callable[[], None]) -> None:
        self._watchers.remove(callback)

    def changed(self) -> None:
        for callback in list(self._watchers):
            callback()


class Button(Control):
    """Pressed, not held."""

    def __init__(self, label: str, on_press: Callable[[], None], *, icon: str | None, tooltip: str):
        super().__init__(label, icon=icon, tooltip=tooltip)
        self.on_press = on_press

    def activate(self, checked: bool) -> None:
        del checked
        self.on_press()


class Toggle(Control):
    """On or off. ``activate`` is the user flipping it and calls ``on_change``;
    ``set`` is the runner moving it and calls nothing, so a runner turning its
    own toggle off never re-enters its own handler."""

    checkable = True

    def __init__(
        self, label: str, on_change: Callable[[bool], None], *, icon: str | None, tooltip: str
    ):
        super().__init__(label, icon=icon, tooltip=tooltip)
        self.on_change = on_change

    def set(self, checked: bool) -> None:
        if checked != self.checked:
            self.checked = checked
            self.changed()

    def activate(self, checked: bool) -> None:
        self.set(checked)
        self.on_change(self.checked)


class RunnerContext(ABC):
    """What a runner can ask of whatever hosts it. Everything registered
    through one context goes away when its program is deselected."""

    @abstractmethod
    def execute(self) -> None:
        """Start a run of the program. Failures are reported, not raised."""

    @abstractmethod
    def on_run_started(self, callback: Callable[[], None]) -> None: ...

    @property
    @abstractmethod
    def params(self) -> BlockMap:
        """The selected program's blocks, the same objects the form edits."""

    @abstractmethod
    def params_written(self) -> None:
        """Announce that ``params`` was changed somewhere other than the form."""

    @abstractmethod
    def request(self, kind: type[Pick], callback: Callable[[Pick | None], None]) -> None:
        """Ask for one ``kind``. A new request cancels the old; ``callback`` gets
        the pick, or None if the request was cancelled."""

    @abstractmethod
    def cancel_request(self) -> None: ...

    @abstractmethod
    def add_control(self, control: Control) -> None: ...

    @abstractmethod
    def status(self, text: str) -> None: ...

    @abstractmethod
    def blockers(self) -> list[str]:
        """What has to be supplied before a run can start; empty when ready."""


@dataclass(frozen=True)
class Runner:
    """One way of starting a program. Declared once, attached per selection."""

    def attach(self, ctx: RunnerContext) -> None:
        raise NotImplementedError
