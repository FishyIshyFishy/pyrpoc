"""Arm, receive what the run needs, then run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyrpoc.src.structs.params import BlockMap, ParameterError
from pyrpoc.src.structs.picks import Pick
from pyrpoc.src.structs.runner import Runner, RunnerContext, Toggle


@dataclass(frozen=True)
class ArmAndRun(Runner):
    """Arm, request a ``pick``, turn what comes back into parameters, run.

    Arming sends the request out and disarming withdraws it; the answer comes
    back from whatever can provide that kind of pick. ``apply`` is the
    program's half: it writes the pick into the blocks it is given, and raises
    ``ParameterError`` when the pick cannot be placed.

    Arming is refused while anything would stop the run starting, because the
    answer is what starts it: an armed control promising a run that cannot
    happen is worse than a refusal that says why.
    """

    pick: type[Pick]
    apply: Callable[[Pick, BlockMap], None]
    label: str = "Arm"
    icon: str | None = None
    tooltip: str = "Arm, then provide what this run needs to start it"

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
            ctx.request(self.pick, on_pick)

        def on_pick(pick: Pick | None) -> None:
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
