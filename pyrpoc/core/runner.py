"""``Runner``: a way of starting a program, with the ``RunnerContext`` it attaches to and the
``Control`` it may offer its user."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pyrpoc.core.program_input import ProgramInputs


class Control:
    """Something a runner offers its user to act on. What it is, and how a host draws it, belong
    to its type."""


class RunnerContext(Protocol):
    """What a runner can ask of whatever hosts it. Everything registered
    through one context goes away when its program is deselected."""

    def execute(self, repeat: bool) -> None:
        """Start a run of the program, which opens one recording. With ``repeat``, the run does
        the program's work again and again into that recording until it is stopped or wound
        down; an edit to its settings starts a new part of the recording at the next pass.
        Failures are reported, not raised."""
        ...

    def on_run_started(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` as each run this context started begins, before
        ``execute`` returns."""
        ...

    def wind_down(self) -> None:
        """Let each run this context started finish the work it is doing, then end."""
        ...

    def on_run_ended(self, callback: Callable[[bool], None]) -> None:
        """Call ``callback(completed)`` as each run this context started ends;
        ``completed`` is False for a run that was stopped or failed, and True for one that
        finished or was wound down."""
        ...

    @property
    def inputs(self) -> ProgramInputs:
        """The program inputs, which a runner may set."""
        ...

    def on_input_changed(self, callback: Callable[[str], None]) -> None:
        """Call ``callback(key)`` after each change to a program input, whoever made it."""
        ...

    def add_control(self, control: Control) -> None:
        """Offer ``control`` to the user, for as long as the program is selected."""
        ...

    def status(self, text: str) -> None: ...


@dataclass(frozen=True)
class Runner:
    """One way of starting a program. Declared once, attached per selection."""

    def attach(self, ctx: RunnerContext) -> None:
        raise NotImplementedError
