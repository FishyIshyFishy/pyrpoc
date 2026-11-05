"""Survey: the region from the stage's corner, tile by tile, keeping focus, measuring motion
and stitching the overview, for a loop to start from."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.plugins.program_io.inputs import (
    EXTRA_LEFT,
    EXTRA_RIGHT,
    SAMPLE_RATE_MHZ,
)

from ..building_blocks.multisite import (
    REGION_INPUTS,
    SURVEY_EMITS,
    SURVEY_FOCUS_INPUTS,
    SURVEY_READS,
    TILE_INPUTS,
    CleanFolder,
    RigMicroscope,
    run_survey,
)
from ..building_blocks.runners import Single


@program_registry.register("survey")
class Survey(Program):
    display_name = "Survey"
    group = "Autonomous"
    order = 31
    devices = [Galvo, DAQ, PriorStage]
    reads = SURVEY_READS
    inputs = [
        *REGION_INPUTS,
        *TILE_INPUTS,
        *SURVEY_FOCUS_INPUTS,
        EXTRA_LEFT,
        EXTRA_RIGHT,
        SAMPLE_RATE_MHZ,
    ]
    cards = {
        "Region": REGION_INPUTS,
        "Tiles": TILE_INPUTS,
        "Focus Search": SURVEY_FOCUS_INPUTS,
        "Scan": (EXTRA_LEFT, EXTRA_RIGHT),
        "DAQ": (SAMPLE_RATE_MHZ,),
    }
    emits = SURVEY_EMITS
    runners = [Single()]
    # Each step of the pipeline starts in a folder holding only the steps before it.
    checks = [CleanFolder(carries_on=False)]

    def run(self, ctx: RunContext) -> None:
        overscan = (ctx.inputs[EXTRA_LEFT], ctx.inputs[EXTRA_RIGHT])
        run_survey(ctx, RigMicroscope(ctx), overscan)
