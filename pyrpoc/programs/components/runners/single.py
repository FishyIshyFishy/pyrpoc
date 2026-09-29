"""Run once."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.structs.runner import Button, Runner, RunnerContext


@dataclass(frozen=True)
class Single(Runner):
    """Run once."""

    def attach(self, ctx: RunnerContext) -> None:
        ctx.add_control(Button("Start", ctx.execute, icon="single", tooltip="Start"))
