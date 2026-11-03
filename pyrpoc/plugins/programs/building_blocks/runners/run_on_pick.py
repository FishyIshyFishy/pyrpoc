"""``RunOnPick``: while on, run each time a pixel is picked into the target point."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.program_inputs import ProgramInput
from pyrpoc.core.runner import Runner, RunnerContext, Toggle

from ..program_inputs import Point


@dataclass(frozen=True)
class RunOnPick(Runner):
    """A toggle that, while on, starts a run each time ``target`` is set to a picked pixel,
    from whichever panel set it."""

    target: ProgramInput[Point]
    label: str
    icon: str | None
    tooltip: str

    def attach(self, ctx: RunnerContext) -> None:
        running = False

        def on_change(checked: bool) -> None:
            if not checked:
                return
            # Refused while anything would stop a run starting, since a pick is what starts it.
            missing = ctx.blockers()
            if missing:
                ctx.status("needs " + ", ".join(missing))
                toggle.set(False)
                return
            ctx.status(
                f"click a pixel in an image panel whose 'Click sets' is {self.target.rule.label}"
            )

        def on_input_changed(key: str) -> None:
            if key != self.target.key or not toggle.checked:
                return
            point: Point = ctx.inputs.get(key)
            # A typed point is not a pick, so editing the volts by hand never moves the beam.
            if not point.picked:
                return
            if running:
                ctx.status("a run is going: the target is set, but no run was started for it")
                return
            ctx.execute()

        def on_run_started() -> None:
            nonlocal running
            running = True

        def on_run_ended(completed: bool) -> None:
            nonlocal running
            del completed
            running = False

        toggle = Toggle(self.label, on_change, icon=self.icon, tooltip=self.tooltip)
        ctx.on_input_changed(on_input_changed)
        ctx.on_run_started(on_run_started)
        ctx.on_run_ended(on_run_ended)
        ctx.add_control(toggle)
