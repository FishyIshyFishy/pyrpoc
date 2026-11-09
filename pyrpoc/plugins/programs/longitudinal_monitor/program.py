"""Longitudinal Monitor: the first loop program. Each unit drawn over the survey is scanned
every period for a duration, treated first if it is in the treated group, and tracked and kept
in focus throughout."""

from __future__ import annotations

from pyrpoc.core.program import RunContext, program_registry
from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.plugins.program_io.inputs import (
    EXTRA_LEFT,
    EXTRA_RIGHT,
    SAMPLE_RATE_MHZ,
    TREATMENT_LINE,
    TREATMENT_PORT,
)

from ..building_blocks.multisite import (
    MONITOR_INPUTS,
    MONITOR_LIVE,
    TREATMENT_INPUTS,
    DigitalLine,
    LoopProgram,
    Microscope,
    RigMicroscope,
    RoutineSpec,
    monitor_routine,
)

RIG_TREATMENT = (*TREATMENT_INPUTS, TREATMENT_PORT, TREATMENT_LINE)


@program_registry.register("longitudinal_monitor")
class LongitudinalMonitor(LoopProgram):
    display_name = "Longitudinal Monitor"
    group = "Autonomous"
    order = 32
    devices = [Galvo, DAQ, PriorStage]
    inputs = [
        *LoopProgram.inputs,
        *MONITOR_INPUTS,
        *RIG_TREATMENT,
        EXTRA_LEFT,
        EXTRA_RIGHT,
        SAMPLE_RATE_MHZ,
    ]
    cards = {
        "Monitor": MONITOR_INPUTS,
        "Treatment": RIG_TREATMENT,
        **LoopProgram.cards,
        "Scan": (EXTRA_LEFT, EXTRA_RIGHT),
        "DAQ": (SAMPLE_RATE_MHZ,),
    }
    live = [*LoopProgram.live, *MONITOR_LIVE]

    def microscope(self, ctx: RunContext) -> Microscope:
        return RigMicroscope(ctx)

    def overscan(self, ctx: RunContext) -> tuple[int, int]:
        return (ctx.inputs[EXTRA_LEFT], ctx.inputs[EXTRA_RIGHT])

    def routines(self, ctx: RunContext) -> tuple[RoutineSpec, ...]:
        line = DigitalLine(ctx.inputs[TREATMENT_PORT], ctx.inputs[TREATMENT_LINE])
        return (monitor_routine(ctx, line),)
