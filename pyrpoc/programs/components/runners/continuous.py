"""Run again and again."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.structs.runner import Runner, RunnerContext, Toggle


@dataclass(frozen=True)
class Continuous(Runner):
    """Start a new run each time one completes, until a run is stopped, fails,
    or is refused. The toggle stays down for as long as it keeps going, and the
    runs form one series, so they add to one library entry per output."""

    def attach(self, ctx: RunnerContext) -> None:
        started = False

        def start() -> None:
            nonlocal started
            started = False
            ctx.execute()
            # execute reports a refusal itself; all that is left is to let go.
            if not started:
                finish()

        def finish() -> None:
            toggle.set(False)
            ctx.close_series()

        def on_change(checked: bool) -> None:
            if checked:
                ctx.open_series()
                start()
            else:
                ctx.close_series()

        def on_run_started() -> None:
            nonlocal started
            started = True

        def on_run_ended(completed: bool) -> None:
            if not toggle.checked:
                return
            if completed:
                start()
            else:
                finish()

        toggle = Toggle("Continuous", on_change, icon="continuous", tooltip="Run until stopped")
        ctx.on_run_started(on_run_started)
        ctx.on_run_ended(on_run_ended)
        ctx.add_control(toggle)
