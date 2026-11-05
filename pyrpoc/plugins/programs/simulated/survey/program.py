"""Simulated Survey: the survey, over the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import EarlierRun, Program, RunContext
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices.simulated import SimulatedScanner
from pyrpoc.plugins.program_io.inputs import (
    DWELL_TIME_US,
    FIELD_PIXELS,
    FIELD_UM,
    FOCUS_FINAL_STEP_UM,
    FOCUS_INITIAL_STEP_UM,
    FOCUS_SEARCH_RANGE_UM,
    SURVEY_HEIGHT_UM,
    SURVEY_WIDTH_UM,
    TILE_OVERLAP,
)
from pyrpoc.plugins.program_io.outputs import Mosaic, SurveyResult

from ...building_blocks.runners import Single
from ...building_blocks.survey.run import SurveySettings, calibration_from, survey
from ..dish_microscope import DishMicroscope

GALVO_CALIBRATION = EarlierRun("simulated_galvo_calibration")


class SimulatedSurvey(Program):
    display_name = "Survey"
    group = "Simulations"
    order = 31
    devices = [SimulatedScanner]
    reads = [GALVO_CALIBRATION]
    sections = [
        InputSection("Region", (SURVEY_WIDTH_UM, SURVEY_HEIGHT_UM)),
        InputSection("Tiles", (FIELD_UM, FIELD_PIXELS, TILE_OVERLAP, DWELL_TIME_US)),
        InputSection(
            "Focus Search", (FOCUS_SEARCH_RANGE_UM, FOCUS_INITIAL_STEP_UM, FOCUS_FINAL_STEP_UM)
        ),
    ]
    emits = {"mosaic": Mosaic, "survey": SurveyResult}
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        calibration = calibration_from(ctx, GALVO_CALIBRATION)
        # The simulated scanner has no mirrors to turn around, so no overscan.
        settings = SurveySettings.from_inputs(ctx.inputs, calibration, (0, 0))
        survey(ctx, DishMicroscope(ctx), settings)
