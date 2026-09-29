"""Runners: the ways a program can be started, declared by the program.

Which entry points make sense depends on the program, so the program declares
them in ``Program.runners`` and whatever hosts it draws and wires them without
knowing what any of them does.

A **runner** is a frozen declaration with one method, ``attach``. Attaching
builds fresh per-program state -- its controls, its callbacks -- against a
``RunnerContext``, so one declaration can be shared as a class attribute.

A ``RunnerContext`` is the host's side: run, stop, the program's parameters, a
place to put controls, and a way to **request** something. A request goes out
("this runner wants a ``PixelPick``") and at most one answer comes back; who
answers is the host's business. A runner that starts a run from a clicked
pixel knows it needs a pixel of a dataset, not that someone was looking at an
image -- the acquisition does not care why the user chose to start it.

**Controls** are Qt-free descriptors of a widget -- a ``Button`` or a
``Toggle`` -- which the host's view renders. They are primitives, not
features: arming is a ``Toggle`` whose callback happens to make a request.

The runners themselves are implementations and live in
``programs/components/runners/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from .params import BlockMap
from .picks import Pick

# --------------------------------------------------------------------------- #
# Controls                                                                     #
# --------------------------------------------------------------------------- #


class Control:
    """A widget a runner asks for. The view renders it and watches it.

    ``icon`` names a file in ``pyrpoc/assets/`` without its extension; the
    label is shown when there is none, and is the tooltip fallback when there is.
    ``enabled`` is the runner's own say: the view also disables every control
    while a run is in progress or something is missing, whatever this holds.
    """

    def __init__(self, label: str, *, icon: str | None = None, tooltip: str = ""):
        self.label = label
        self.icon = icon
        self.tooltip = tooltip
        self._enabled = True
        self._watchers: list[Callable[[], None]] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        value = bool(value)
        if value != self._enabled:
            self._enabled = value
            self.changed()

    def watch(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` whenever this control's state changes."""
        self._watchers.append(callback)

    def unwatch(self, callback: Callable[[], None]) -> None:
        if callback in self._watchers:
            self._watchers.remove(callback)

    def changed(self) -> None:
        for callback in list(self._watchers):
            callback()


class Button(Control):
    """Pressed, not held. ``press`` is what the view calls on a click."""

    def __init__(
        self,
        label: str,
        on_press: Callable[[], None],
        *,
        icon: str | None = None,
        tooltip: str = "",
    ):
        super().__init__(label, icon=icon, tooltip=tooltip)
        self.on_press = on_press

    def press(self) -> None:
        self.on_press()


class Toggle(Control):
    """On or off.

    Two ways in, deliberately different: ``toggle`` is the user flipping it,
    which calls ``on_change``; ``set`` is the runner putting it somewhere,
    which updates the view and calls nothing. A runner turning its own toggle
    off after a pick must not re-enter its own off handler.
    """

    def __init__(
        self,
        label: str,
        on_change: Callable[[bool], None],
        *,
        icon: str | None = None,
        tooltip: str = "",
    ):
        super().__init__(label, icon=icon, tooltip=tooltip)
        self.on_change = on_change
        self.checked = False

    def set(self, checked: bool) -> None:
        checked = bool(checked)
        if checked != self.checked:
            self.checked = checked
            self.changed()

    def toggle(self, checked: bool) -> None:
        self.set(checked)
        self.on_change(self.checked)


# --------------------------------------------------------------------------- #
# The host's side                                                              #
# --------------------------------------------------------------------------- #


class RunnerContext(ABC):
    """What a runner can ask of whatever hosts it.

    One context per attachment: everything a runner registers through it --
    controls, run callbacks, a pending pick -- goes away when the program it
    was attached for is deselected.
    """

    # -- running ----------------------------------------------------------- #

    @abstractmethod
    def execute(self, continuous: bool = False) -> None:
        """Start the selected program. Failures are reported, not raised."""

    @abstractmethod
    def stop(self) -> None: ...

    @property
    @abstractmethod
    def running(self) -> bool: ...

    @abstractmethod
    def on_run_started(self, callback: Callable[[], None]) -> None: ...

    @abstractmethod
    def on_run_finished(self, callback: Callable[[], None]) -> None: ...

    # -- parameters -------------------------------------------------------- #

    @property
    @abstractmethod
    def params(self) -> BlockMap:
        """The selected program's blocks, the same objects the form edits."""

    @abstractmethod
    def params_written(self) -> None:
        """Announce that ``params`` was changed somewhere other than the form."""

    # -- requests ---------------------------------------------------------- #

    @abstractmethod
    def request(self, kind: type[Pick], callback: Callable[[Pick | None], None]) -> None:
        """Announce that this runner wants one ``kind``, and wait for it.

        One request at a time: a new one cancels the old. ``callback`` gets
        what was provided, or ``None`` if the request was cancelled.
        """

    @abstractmethod
    def cancel_request(self) -> None: ...

    # -- view -------------------------------------------------------------- #

    @abstractmethod
    def add_control(self, control: Control) -> None: ...

    @abstractmethod
    def status(self, text: str) -> None: ...

    @abstractmethod
    def blockers(self) -> list[str]:
        """What has to be supplied before a run can start; empty when ready."""


# --------------------------------------------------------------------------- #
# The base                                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Runner:
    """One way of starting a program. Declared once, attached per selection."""

    def attach(self, ctx: RunnerContext) -> None:
        raise NotImplementedError
