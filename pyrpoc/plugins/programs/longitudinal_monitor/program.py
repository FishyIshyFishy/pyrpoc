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
from pyrpoc.plugins.program_io.requests import RUN_NOW

from ..building_blocks.multisite import (
    LOOP_EMITS,
    LOOP_FOCUS_INPUTS,
    LOOP_READS,
    MONITOR_INPUTS,
    MONITOR_LIVE,
    STEERING,
    TREATMENT_INPUTS,
    CleanFolder,
    DigitalLine,
    RigMicroscope,
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
        *STEERING,
        *MONITOR_INPUTS,
        *RIG_TREATMENT,
        *LOOP_FOCUS_INPUTS,
        EXTRA_LEFT,
        EXTRA_RIGHT,
        SAMPLE_RATE_MHZ,
    ]
    # The units and pauses have no row: they are drawn and set in the Autonomy panel.
    cards = {
        "Monitor": MONITOR_INPUTS,
        "Treatment": RIG_TREATMENT,
        "Focus": LOOP_FOCUS_INPUTS,
        "Scan": (EXTRA_LEFT, EXTRA_RIGHT),
        "DAQ": (SAMPLE_RATE_MHZ,),
    }
    live = [*STEERING, *MONITOR_LIVE]
    requests = [RUN_NOW]
    emits = LOOP_EMITS
    runners = [Single()]
    # Each step of the pipeline starts in a folder holding only the steps before it.
    checks = [CleanFolder()]

    def run(self, ctx: RunContext) -> None:
        overscan = (ctx.inputs[EXTRA_LEFT], ctx.inputs[EXTRA_RIGHT])
        line = DigitalLine(ctx.inputs[TREATMENT_PORT], ctx.inputs[TREATMENT_LINE])
        run_loop(ctx, RigMicroscope(ctx), overscan, (monitor_routine(ctx, line),))
