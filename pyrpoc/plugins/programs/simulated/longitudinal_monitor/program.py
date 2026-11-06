"""Simulated Longitudinal Monitor: the loop, over the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext
from pyrpoc.plugins.devices.simulated import SimulatedScanner

from ...building_blocks.multisite import (
    LOOP_EMITS,
    LOOP_FOCUS_INPUTS,
    LOOP_READS,
    LOOP_UNITS,
    MONITOR_INPUTS,
    MONITOR_LIVE,
    TREATMENT_INPUTS,
    CalibrationMeasured,
    CleanFolder,
    DigitalLine,
    StartingUnits,
    SurveyFinished,
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
    inputs = [*LOOP_UNITS, *MONITOR_INPUTS, *TREATMENT_INPUTS, *LOOP_FOCUS_INPUTS]
    # The units have no row: they are drawn over the survey. The
    # dish's treatment light answers any line, so the line is not asked for.
    cards = {"Monitor": MONITOR_INPUTS, "Treatment": TREATMENT_INPUTS, "Focus": LOOP_FOCUS_INPUTS}
    live = [*LOOP_UNITS, *MONITOR_LIVE]
    emits = LOOP_EMITS
    runners = [Single()]
    # Each step of the pipeline starts in a folder holding only the steps before it, and a loop
    # cut short may be carried on in its own folder.
    checks = [
        CleanFolder(carries_on=True),
        CalibrationMeasured(),
        SurveyFinished(),
        StartingUnits(),
    ]

    def run(self, ctx: RunContext) -> None:
        routine = monitor_routine(ctx, DigitalLine(0, 0))
        run_loop(ctx, DishMicroscope(ctx), (0, 0), (routine,))
