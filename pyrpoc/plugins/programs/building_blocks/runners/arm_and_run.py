"""``ArmAndRun``: arm, receive a picked pixel, then run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyrpoc.core.program_input import ParameterError, ProgramInputs
from pyrpoc.core.runner import Runner, RunnerContext
from pyrpoc.plugins.program_io.inputs import PICK, PixelPick
from pyrpoc.utils.controls import Toggle


@dataclass(frozen=True)
class ArmAndRun(Runner):
    """Arm, wait for the next pixel picked on an image, write it into the program inputs
    with ``apply``, then run."""

    # The program's half: writes the pixel into the program inputs, raising
    # ``ParameterError`` when the pixel cannot be placed.
    apply: Callable[[PixelPick, ProgramInputs], None]
    label: str
    icon: str | None
    tooltip: str

    def attach(self, ctx: RunnerContext) -> None:
        def on_change(checked: bool) -> None:
            ctx.status("waiting for a pick on a display" if checked else "ready")

        def on_input_changed(key: str) -> None:
            if key != PICK.key or not toggle.checked:
                return
            toggle.set(False)
            try:
                self.apply(ctx.inputs.get(PICK.key), ctx.inputs)
            except ParameterError as exc:
                ctx.status(str(exc))
                return
            ctx.execute(repeat=False)

        def on_run_started() -> None:
            # A run started some other way disarms, so a later click does not start another.
            toggle.set(False)

        toggle = Toggle(self.label, on_change, icon=self.icon, tooltip=self.tooltip)
        ctx.on_input_changed(on_input_changed)
        ctx.on_run_started(on_run_started)
        ctx.add_control(toggle)
