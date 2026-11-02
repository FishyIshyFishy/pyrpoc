"""Arm, receive what the run needs, then run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyrpoc.structs.data_library.data import Pixel
from pyrpoc.structs.data_library.dataset import Dataset
from pyrpoc.structs.plugins.params import BlockMap, ParameterError
from pyrpoc.structs.plugins.programs.runner import Runner, RunnerContext, Toggle


@dataclass(frozen=True)
class ArmAndRun(Runner):
    """Arm, request a pixel, turn what comes back into parameters, run.

    ``apply`` is the program's half: it writes the pixel into the blocks, raising
    ``ParameterError`` when the pixel cannot be placed. Arming is refused while
    anything would stop the run starting, since the answer is what starts it.
    """

    apply: Callable[[Dataset[Pixel], BlockMap], None]
    label: str
    icon: str | None
    tooltip: str

    def attach(self, ctx: RunnerContext) -> None:
        def on_change(checked: bool) -> None:
            if not checked:
                ctx.cancel_request()
                return
            missing = ctx.blockers()
            if missing:
                ctx.status("needs " + ", ".join(missing))
                toggle.set(False)
                return
            ctx.request_pixel(on_pick)

        def on_pick(pick: Dataset[Pixel] | None) -> None:
            toggle.set(False)
            if pick is None:
                return
            try:
                self.apply(pick, ctx.params)
            except ParameterError as exc:
                ctx.status(str(exc))
                return
            ctx.params_written()
            ctx.execute()

        def on_run_started() -> None:
            if toggle.checked:
                ctx.cancel_request()

        toggle = Toggle(self.label, on_change, icon=self.icon, tooltip=self.tooltip)
        ctx.on_run_started(on_run_started)
        ctx.add_control(toggle)
