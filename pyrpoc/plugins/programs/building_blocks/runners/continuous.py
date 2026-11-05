"""Run again and again."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.runner import Runner, RunnerContext
from pyrpoc.utils.controls import Toggle


@dataclass(frozen=True)
class Continuous(Runner):
    """One run that repeats the program's work into one recording until the toggle goes up, or
    the run is stopped, fails or is refused. An edit to what it runs with starts a new part of
    that recording at its next pass."""

    def attach(self, ctx: RunnerContext) -> None:
        started = False

        def begin() -> None:
            nonlocal started
            started = False
            ctx.execute(repeat=True)
            # execute reports a refusal itself; all that is left is to let go.
            if not started:
                toggle.set(False)

        def on_change(checked: bool) -> None:
            if checked:
                begin()
            else:
                ctx.wind_down()

        def on_run_started() -> None:
            nonlocal started
            started = True

        def on_run_ended(_completed: bool) -> None:
            toggle.set(False)

        toggle = Toggle("Continuous", on_change, icon="continuous", tooltip="Run until stopped")
        ctx.on_run_started(on_run_started)
        ctx.on_run_ended(on_run_ended)
        ctx.add_control(toggle)
