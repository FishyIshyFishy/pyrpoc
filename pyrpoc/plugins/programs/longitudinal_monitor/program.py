"""Longitudinal Monitor: the first loop program. Each unit drawn over the survey is scanned
every period for a duration, treated first if it is in the treated group, and tracked and kept
in focus throughout."""

from __future__ import annotations

from pyrpoc.core.program import EarlierRun, Program, RunContext, program_registry
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.plugins.program_io.inputs import (
    DURATION_MIN,
    DWELL_TIME_US,
    EXTRA_LEFT,
    EXTRA_RIGHT,
    FOCUS_CHECK_S,
    FOCUS_FINAL_STEP_UM,
    PERIOD_S,
    SAMPLE_RATE_MHZ,
    TREAT,
    TREATED_GROUP,
    TREATMENT_LINE,
    TREATMENT_PORT,
    UNITS,
)
from pyrpoc.plugins.program_io.outputs import LoopEvent, LoopState, LoopUnits, UnitFrame

from ..building_blocks.loop.monitor import run_monitor
from ..building_blocks.loop.steering import STEERING
from ..building_blocks.microscope import DigitalLine
from ..building_blocks.rig import RigMicroscope
from ..building_blocks.runners import Single

SURVEY = EarlierRun("survey")


@program_registry.register("longitudinal_monitor")
class LongitudinalMonitor(Program):
    display_name = "Longitudinal Monitor"
    group = "Autonomous"
    order = 32
    devices = [Galvo, DAQ, PriorStage]
    reads = [SURVEY]
    sections = [
        InputSection("Monitor", (UNITS, PERIOD_S, DURATION_MIN, DWELL_TIME_US)),
        InputSection("Treatment", (TREAT, TREATED_GROUP, TREATMENT_PORT, TREATMENT_LINE)),
        InputSection("Focus", (FOCUS_CHECK_S, FOCUS_FINAL_STEP_UM)),
        InputSection("Scan", (EXTRA_LEFT, EXTRA_RIGHT)),
        InputSection("DAQ", (SAMPLE_RATE_MHZ,)),
    ]
    live = [PERIOD_S, DURATION_MIN, *STEERING]
    emits = {"log": LoopEvent, "frames": UnitFrame, "units": LoopUnits, "state": LoopState}
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        overscan = (ctx.inputs[EXTRA_LEFT], ctx.inputs[EXTRA_RIGHT])
        line = DigitalLine(ctx.inputs[TREATMENT_PORT], ctx.inputs[TREATMENT_LINE])
        run_monitor(ctx, RigMicroscope(ctx), SURVEY, overscan, line)
