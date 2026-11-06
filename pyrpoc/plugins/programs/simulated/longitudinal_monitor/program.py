"""Simulated Longitudinal Monitor: the loop, over the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import RunContext
from pyrpoc.plugins.devices.simulated import SimulatedScanner

from ...building_blocks.multisite import (
    MONITOR_INPUTS,
    MONITOR_LIVE,
    TREATMENT_INPUTS,
    DigitalLine,
    LoopProgram,
    Microscope,
    RoutineSpec,
    monitor_routine,
)
from ..dish_microscope import DishMicroscope


class SimulatedLongitudinalMonitor(LoopProgram):
    display_name = "Longitudinal Monitor"
    group = "Simulations"
    order = 32
    devices = [SimulatedScanner]
    inputs = [*LoopProgram.inputs, *MONITOR_INPUTS, *TREATMENT_INPUTS]
    cards = {"Monitor": MONITOR_INPUTS, "Treatment": TREATMENT_INPUTS, **LoopProgram.cards}
    live = [*LoopProgram.live, *MONITOR_LIVE]

    def microscope(self, ctx: RunContext) -> Microscope:
        return DishMicroscope(ctx)

    def overscan(self, ctx: RunContext) -> tuple[int, int]:
        # The simulated scanner has no mirrors to turn around.
        return (0, 0)

    def routines(self, ctx: RunContext) -> tuple[RoutineSpec, ...]:
        # The dish's treatment light answers any line, so the line is not asked for.
        return (monitor_routine(ctx, DigitalLine(0, 0)),)
