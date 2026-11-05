"""Simulated Longitudinal Monitor: the loop, over the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext
from pyrpoc.plugins.devices.simulated import SimulatedScanner
from pyrpoc.plugins.program_io.requests import RUN_NOW

from ...building_blocks.multisite import (
    LOOP_EMITS,
    LOOP_FOCUS_INPUTS,
    LOOP_READS,
    MONITOR_INPUTS,
    MONITOR_LIVE,
    STEERING,
    TREATMENT_INPUTS,
    DigitalLine,
    monitor_routine,
    run_loop,
)
from ...building_blocks.runners import Single
from ..dish_microscope import DishMicroscope


class SimulatedLongitudinalMonitor(Program):
    display_name = "Longitudinal Monitor"
    group = "Simulations"
    order = 32
    devices = [SimulatedScanner]
    reads = LOOP_READS
    inputs = [*STEERING, *MONITOR_INPUTS, *TREATMENT_INPUTS, *LOOP_FOCUS_INPUTS]
    # The units and pauses have no row: they are drawn and set in the Autonomy panel. The
    # dish's treatment light answers any line, so the line is not asked for.
    cards = {"Monitor": MONITOR_INPUTS, "Treatment": TREATMENT_INPUTS, "Focus": LOOP_FOCUS_INPUTS}
    live = [*STEERING, *MONITOR_LIVE]
    requests = [RUN_NOW]
    emits = LOOP_EMITS
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        routine = monitor_routine(ctx, DigitalLine(0, 0))
        run_loop(ctx, DishMicroscope(ctx), (0, 0), (routine,))
