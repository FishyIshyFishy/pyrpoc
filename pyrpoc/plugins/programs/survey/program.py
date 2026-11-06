"""Survey: the region from the stage's corner, tile by tile, keeping focus, measuring motion
and stitching the overview, for a loop to start from."""

from __future__ import annotations

from pyrpoc.core.program import RunContext, program_registry
from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.plugins.program_io.inputs import EXTRA_LEFT, EXTRA_RIGHT, SAMPLE_RATE_MHZ

from ..building_blocks.multisite import Microscope, RigMicroscope, SurveyProgram


@program_registry.register("survey")
class Survey(SurveyProgram):
    display_name = "Survey"
    group = "Autonomous"
    order = 31
    devices = [Galvo, DAQ, PriorStage]
    inputs = [*SurveyProgram.inputs, EXTRA_LEFT, EXTRA_RIGHT, SAMPLE_RATE_MHZ]
    cards = {
        **SurveyProgram.cards,
        "Scan": (EXTRA_LEFT, EXTRA_RIGHT),
        "DAQ": (SAMPLE_RATE_MHZ,),
    }

    def microscope(self, ctx: RunContext) -> Microscope:
        return RigMicroscope(ctx)

    def overscan(self, ctx: RunContext) -> tuple[int, int]:
        return (ctx.inputs[EXTRA_LEFT], ctx.inputs[EXTRA_RIGHT])
