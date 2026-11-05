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
from pyrpoc.plugins.program_io.outputs import GalvoCalibrationOutput

from ..building_blocks.multisite.bindings.rig import RigMicroscope
from ..building_blocks.multisite.stages.survey.walk import (
    FOCUS_INPUTS,
    REGION_INPUTS,
    SURVEY_EMITS,
    TILE_INPUTS,
    SurveySettings,
    survey,
)
from ..building_blocks.runners import Single


@program_registry.register("survey")
class Survey(Program):
    display_name = "Survey"
    group = "Autonomous"
    order = 31
    devices = [Galvo, DAQ, PriorStage]
    reads = {"calibration": GalvoCalibrationOutput}
    inputs = [*REGION_INPUTS, *TILE_INPUTS, *FOCUS_INPUTS, EXTRA_LEFT, EXTRA_RIGHT, SAMPLE_RATE_MHZ]
    cards = {
        "Region": REGION_INPUTS,
        "Tiles": TILE_INPUTS,
        "Focus Search": FOCUS_INPUTS,
        "Scan": (EXTRA_LEFT, EXTRA_RIGHT),
        "DAQ": (SAMPLE_RATE_MHZ,),
    }
    emits = SURVEY_EMITS
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        calibration = ctx.earlier("calibration", GalvoCalibrationOutput).measured()
        overscan = (ctx.inputs[EXTRA_LEFT], ctx.inputs[EXTRA_RIGHT])
        settings = SurveySettings.from_inputs(ctx.inputs, calibration, overscan)
        survey(ctx, RigMicroscope(ctx), settings)
