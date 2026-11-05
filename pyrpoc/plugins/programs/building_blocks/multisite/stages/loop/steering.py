"""Steering a loop between its steps: units added or removed in the Autonomy panel and units
paused or resumed, read as live program inputs, and units asked to run now, taken as
requests."""

# The loop logs what it did about each change, not the change itself.

from __future__ import annotations

from typing import Any

from pyrpoc.core.program import RunContext
from pyrpoc.core.program_input import ProgramInput
from pyrpoc.plugins.program_io.inputs import PAUSED_UNITS, UNITS, Unit
from pyrpoc.plugins.program_io.requests import RUN_NOW

# Every loop program declares these live, and accepts RUN_NOW, so the panel can steer it.
STEERING: list[ProgramInput[Any]] = [UNITS, PAUSED_UNITS]


class Steering:
    """What the panel has asked for since the loop last looked."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    def units(self) -> tuple[Unit, ...]:
        return self.ctx.current(UNITS)

    def paused(self) -> frozenset[int]:
        return frozenset(self.ctx.current(PAUSED_UNITS))

    def run_now(self) -> list[int]:
        """The units asked to run now since the loop last looked."""
        return self.ctx.requests(RUN_NOW)
