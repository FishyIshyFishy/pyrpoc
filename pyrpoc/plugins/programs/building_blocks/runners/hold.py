"""Run a stand-in in the program's place, such as holding lines high instead of scanning."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.structs.plugins.programs.program import Program
from pyrpoc.structs.plugins.programs.runner import Runner, RunnerContext, Toggle


@dataclass(frozen=True)
class Hold(Runner):
    """A toggle that starts ``stand_in`` and stays down while it runs. The view
    locks every control during a run, so Stop is what ends it, as for any run."""

    stand_in: type[Program]
    label: str
    tooltip: str

    def attach(self, ctx: RunnerContext) -> None:
        started = False

        def on_change(checked: bool) -> None:
            nonlocal started
            if not checked:
                return
            started = False
            ctx.execute(self.stand_in)
            # execute reports a refusal itself; all that is left is to let go.
            if not started:
                toggle.set(False)

        def on_run_started() -> None:
            nonlocal started
            started = True

        def on_run_ended(completed: bool) -> None:
            del completed
            toggle.set(False)

        toggle = Toggle(self.label, on_change, icon=None, tooltip=self.tooltip)
        ctx.on_run_started(on_run_started)
        ctx.on_run_ended(on_run_ended)
        ctx.add_control(toggle)
