"""``Runner``: a way of starting a program, with the ``RunnerContext`` it
attaches to and the controls it asks to be drawn."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.core.program_input import ProgramInputs


class Control:
    """A widget a runner asks for, described without Qt. ``icon`` names a file
    in ``utils/assets/`` without its extension."""

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
        """Respond to the user clicking it."""
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
    ``set`` is the runner moving it and calls nothing."""

    checkable = True

    def __init__(
        self, label: str, on_change: Callable[[bool], None], *, icon: str | None, tooltip: str
    ):
        super().__init__(label, icon=icon, tooltip=tooltip)
        self.on_change = on_change

    def set(self, checked: bool) -> None:
        # Calls nothing, so a runner turning its own toggle off never re-enters
        # its handler.
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
    def execute(self, repeat: bool) -> None:
        """Start a run of the program, which opens one recording. With ``repeat``, the run does
        the program's work again and again into that recording until it is stopped or wound
        down; it also winds down by itself once anything it was started with changes, so
        edits reach a new recording. Failures are reported, not raised."""

    @abstractmethod
    def on_run_started(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` as each run this context started begins, before
        ``execute`` returns."""

    @abstractmethod
    def wind_down(self) -> None:
        """Let each run this context started finish the work it is doing, then end."""

    @abstractmethod
    def on_run_ended(self, callback: Callable[[bool], None]) -> None:
        """Call ``callback(completed)`` as each run this context started ends;
        ``completed`` is False for a run that was stopped or failed, and True for one that
        finished or was wound down."""

    @property
    @abstractmethod
    def inputs(self) -> ProgramInputs:
        """The program inputs, which a runner may set."""

    @abstractmethod
    def on_input_changed(self, callback: Callable[[str], None]) -> None:
        """Call ``callback(key)`` after each change to a program input, whoever made it."""

    @abstractmethod
    def add_control(self, control: Control) -> None: ...

    @abstractmethod
    def status(self, text: str) -> None: ...


@dataclass(frozen=True)
class Runner:
    """One way of starting a program. Declared once, attached per selection."""

    def attach(self, ctx: RunnerContext) -> None:
        raise NotImplementedError
