"""Run once."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.runner import Runner, RunnerContext
from pyrpoc.utils.controls import Button


@dataclass(frozen=True)
class Single(Runner):
    """Run once."""

    def attach(self, ctx: RunnerContext) -> None:
        ctx.add_control(
            Button("Start", lambda: ctx.execute(repeat=False), icon="single", tooltip="Start")
        )
