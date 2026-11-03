"""``ArmAndRun``: arm, receive a picked pixel, then run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyrpoc.core.input_requests import PixelPick
from pyrpoc.core.program_inputs import ParameterError, ProgramInputs
from pyrpoc.core.runner import Runner, RunnerContext, Toggle


@dataclass(frozen=True)
class ArmAndRun(Runner):
    """Arm, ask the panels for a pixel, write it into the parameters with
    ``apply``, then run."""

    # The program's half: writes the pixel into the program inputs, raising
    # ``ParameterError`` when the pixel cannot be placed.
    apply: Callable[[PixelPick, ProgramInputs], None]
    label: str
    icon: str | None
    tooltip: str

    def attach(self, ctx: RunnerContext) -> None:
        def on_change(checked: bool) -> None:
            if not checked:
                ctx.cancel_request()
                return
            # Refused while anything would stop the run starting, since the
            # answer is what starts it.
            missing = ctx.blockers()
            if missing:
                ctx.status("needs " + ", ".join(missing))
                toggle.set(False)
                return
            ctx.request(PixelPick, "waiting for a pick on a display", on_pick)

        def on_pick(pick: PixelPick | None) -> None:
            toggle.set(False)
            if pick is None:
                return
            try:
                self.apply(pick, ctx.inputs)
            except ParameterError as exc:
                ctx.status(str(exc))
                return
            ctx.execute()

        def on_run_started() -> None:
            if toggle.checked:
                ctx.cancel_request()

        toggle = Toggle(self.label, on_change, icon=self.icon, tooltip=self.tooltip)
        ctx.on_run_started(on_run_started)
        ctx.add_control(toggle)
