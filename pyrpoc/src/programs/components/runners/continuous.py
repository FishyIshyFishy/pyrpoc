"""Run until stopped."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.src.structs.runner import Button, Runner, RunnerContext


@dataclass(frozen=True)
class Continuous(Runner):
    """Run until stopped."""

    def attach(self, ctx: RunnerContext) -> None:
        ctx.add_control(
            Button(
                "Continuous",
                lambda: ctx.execute(continuous=True),
                icon="multi",
                tooltip="Continuous acquisition",
            )
        )
