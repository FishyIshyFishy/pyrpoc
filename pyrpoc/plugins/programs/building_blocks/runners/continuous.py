"""Run again and again."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.runner import Runner, RunnerContext
from pyrpoc.utils.controls import Toggle


@dataclass(frozen=True)
class Continuous(Runner):
    """One run that repeats the program's work into one recording until the toggle goes up, a
    run is stopped or fails, or one is refused. An edit to what it runs with ends that run, and
    the next starts a new recording."""

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

        def on_run_ended(completed: bool) -> None:
            # A run that ended while the toggle is still down was wound down by an edit.
            if toggle.checked and completed:
                begin()
            else:
                toggle.set(False)

        toggle = Toggle("Continuous", on_change, icon="continuous", tooltip="Run until stopped")
        ctx.on_run_started(on_run_started)
        ctx.on_run_ended(on_run_ended)
        ctx.add_control(toggle)
