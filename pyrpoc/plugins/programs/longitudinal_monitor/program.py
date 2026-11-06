"""Longitudinal Monitor: the first loop program. Each unit drawn over the survey is scanned
every period for a duration, treated first if it is in the treated group, and tracked and kept
in focus throughout."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.plugins.program_io.inputs import (
    EXTRA_LEFT,
    EXTRA_RIGHT,
    SAMPLE_RATE_MHZ,
    TREATMENT_LINE,
    TREATMENT_PORT,
)

from ..building_blocks.multisite import (
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
    RigMicroscope,
    StartingUnits,
    SurveyFinished,
    monitor_routine,
    run_loop,
)
from ..building_blocks.runners import Single

RIG_TREATMENT = (*TREATMENT_INPUTS, TREATMENT_PORT, TREATMENT_LINE)


@program_registry.register("longitudinal_monitor")
class LongitudinalMonitor(Program):
    display_name = "Longitudinal Monitor"
    group = "Autonomous"
    order = 32
    devices = [Galvo, DAQ, PriorStage]
    reads = LOOP_READS
    inputs = [
        *LOOP_UNITS,
        *MONITOR_INPUTS,
        *RIG_TREATMENT,
        *LOOP_FOCUS_INPUTS,
        EXTRA_LEFT,
        EXTRA_RIGHT,
        SAMPLE_RATE_MHZ,
    ]
    # The units have no row: they are drawn over the survey.
    cards = {
        "Monitor": MONITOR_INPUTS,
        "Treatment": RIG_TREATMENT,
        "Focus": LOOP_FOCUS_INPUTS,
        "Scan": (EXTRA_LEFT, EXTRA_RIGHT),
        "DAQ": (SAMPLE_RATE_MHZ,),
    }
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
        overscan = (ctx.inputs[EXTRA_LEFT], ctx.inputs[EXTRA_RIGHT])
        line = DigitalLine(ctx.inputs[TREATMENT_PORT], ctx.inputs[TREATMENT_LINE])
        run_loop(ctx, RigMicroscope(ctx), overscan, (monitor_routine(ctx, line),))
