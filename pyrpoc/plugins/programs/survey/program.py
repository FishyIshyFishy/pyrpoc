"""Survey: the region from the stage's corner, tile by tile, keeping focus, measuring motion
and stitching the overview, for a loop to start from."""

from __future__ import annotations

from pyrpoc.core.program import EarlierRun, Program, RunContext, program_registry
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.plugins.program_io.inputs import (
    DWELL_TIME_US,
    EXTRA_LEFT,
    EXTRA_RIGHT,
    FIELD_PIXELS,
    FIELD_UM,
    FOCUS_FINAL_STEP_UM,
    FOCUS_INITIAL_STEP_UM,
    FOCUS_SEARCH_RANGE_UM,
    SAMPLE_RATE_MHZ,
    SURVEY_HEIGHT_UM,
    SURVEY_WIDTH_UM,
    TILE_OVERLAP,
)
from pyrpoc.plugins.program_io.outputs import Mosaic, SurveyResult

from ..building_blocks.rig import RigMicroscope
from ..building_blocks.runners import Single
from ..building_blocks.survey.run import SurveySettings, calibration_from, survey

GALVO_CALIBRATION = EarlierRun("galvo_calibration")


@program_registry.register("survey")
class Survey(Program):
    display_name = "Survey"
    group = "Autonomous"
    order = 31
    devices = [Galvo, DAQ, PriorStage]
    reads = [GALVO_CALIBRATION]
    sections = [
        InputSection("Region", (SURVEY_WIDTH_UM, SURVEY_HEIGHT_UM)),
        InputSection("Tiles", (FIELD_UM, FIELD_PIXELS, TILE_OVERLAP, DWELL_TIME_US)),
        InputSection(
            "Focus Search", (FOCUS_SEARCH_RANGE_UM, FOCUS_INITIAL_STEP_UM, FOCUS_FINAL_STEP_UM)
        ),
        InputSection("Scan", (EXTRA_LEFT, EXTRA_RIGHT)),
        InputSection("DAQ", (SAMPLE_RATE_MHZ,)),
    ]
    emits = {"mosaic": Mosaic, "survey": SurveyResult}
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        calibration = calibration_from(ctx, GALVO_CALIBRATION)
        overscan = (ctx.inputs[EXTRA_LEFT], ctx.inputs[EXTRA_RIGHT])
        settings = SurveySettings.from_inputs(ctx.inputs, calibration, overscan)
        survey(ctx, RigMicroscope(ctx), settings)
