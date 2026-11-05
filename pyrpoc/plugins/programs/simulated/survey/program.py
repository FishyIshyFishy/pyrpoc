"""Simulated Survey: the survey, over the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext
from pyrpoc.plugins.devices.simulated import SimulatedScanner
from pyrpoc.plugins.program_io.outputs import GalvoCalibrationOutput

from ...building_blocks.multisite.stages.survey.walk import (
    FOCUS_INPUTS,
    REGION_INPUTS,
    SURVEY_EMITS,
    TILE_INPUTS,
    SurveySettings,
    survey,
)
from ...building_blocks.runners import Single
from ..dish_microscope import DishMicroscope


class SimulatedSurvey(Program):
    display_name = "Survey"
    group = "Simulations"
    order = 31
    devices = [SimulatedScanner]
    reads = {"calibration": GalvoCalibrationOutput}
    inputs = [*REGION_INPUTS, *TILE_INPUTS, *FOCUS_INPUTS]
    cards = {"Region": REGION_INPUTS, "Tiles": TILE_INPUTS, "Focus Search": FOCUS_INPUTS}
    emits = SURVEY_EMITS
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        calibration = ctx.earlier("calibration", GalvoCalibrationOutput).measured()
        # The simulated scanner has no mirrors to turn around, so no overscan.
        settings = SurveySettings.from_inputs(ctx.inputs, calibration, (0, 0))
        survey(ctx, DishMicroscope(ctx), settings)
