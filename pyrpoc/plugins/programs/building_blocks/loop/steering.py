"""Steering a loop between its steps: units added or removed in the Autonomy panel, units
paused or resumed, and units asked to run now, all read as live program inputs."""

# The loop logs what it did about each change, not the change itself.

from __future__ import annotations

from typing import Any

from pyrpoc.core.program import RunContext
from pyrpoc.core.program_input import ProgramInput
from pyrpoc.plugins.program_io.inputs import PAUSED_UNITS, RUN_NOW, UNITS, Unit

# Every loop program declares these live, so the panel can steer it.
STEERING: list[ProgramInput[Any]] = [UNITS, PAUSED_UNITS, RUN_NOW]


class Steering:
    """What the panel has asked for since the loop last looked."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.answered = max((request.serial for request in ctx.current(RUN_NOW)), default=0)

    def units(self) -> tuple[Unit, ...]:
        return self.ctx.current(UNITS)

    def paused(self) -> frozenset[int]:
        return frozenset(self.ctx.current(PAUSED_UNITS))

    def run_now(self) -> list[int]:
        """The units asked to run now that the loop has not yet answered."""
        fresh = [request for request in self.ctx.current(RUN_NOW) if request.serial > self.answered]
        if fresh:
            self.answered = max(request.serial for request in fresh)
        return [request.unit for request in fresh]
