"""Simulated Longitudinal Monitor: the loop, over the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import EarlierRun, Program, RunContext
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices.simulated import SimulatedScanner
from pyrpoc.plugins.program_io.inputs import (
    DURATION_MIN,
    DWELL_TIME_US,
    FOCUS_CHECK_S,
    FOCUS_FINAL_STEP_UM,
    PERIOD_S,
    TREAT,
    TREATED_GROUP,
    UNITS,
)
from pyrpoc.plugins.program_io.outputs import LoopEvent, LoopState, LoopUnits, UnitFrame

from ...building_blocks.loop.monitor import run_monitor
from ...building_blocks.loop.steering import STEERING
from ...building_blocks.microscope import DigitalLine
from ...building_blocks.runners import Single
from ..dish_microscope import DishMicroscope

SURVEY = EarlierRun("simulated_survey")


class SimulatedLongitudinalMonitor(Program):
    display_name = "Longitudinal Monitor"
    group = "Simulations"
    order = 32
    devices = [SimulatedScanner]
    reads = [SURVEY]
    sections = [
        InputSection("Monitor", (UNITS, PERIOD_S, DURATION_MIN, DWELL_TIME_US)),
        # The dish's treatment light answers any line, so the line is not asked for.
        InputSection("Treatment", (TREAT, TREATED_GROUP)),
        InputSection("Focus", (FOCUS_CHECK_S, FOCUS_FINAL_STEP_UM)),
    ]
    live = [PERIOD_S, DURATION_MIN, *STEERING]
    emits = {"log": LoopEvent, "frames": UnitFrame, "units": LoopUnits, "state": LoopState}
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        run_monitor(ctx, DishMicroscope(ctx), SURVEY, (0, 0), DigitalLine(0, 0))
