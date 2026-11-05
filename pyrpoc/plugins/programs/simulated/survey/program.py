"""Simulated Survey: the survey, over the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext
from pyrpoc.plugins.devices.simulated import SimulatedScanner

from ...building_blocks.multisite import (
    REGION_INPUTS,
    SURVEY_EMITS,
    SURVEY_FOCUS_INPUTS,
    SURVEY_READS,
    TILE_INPUTS,
    CleanFolder,
    run_survey,
)
from ...building_blocks.runners import Single
from ..dish_microscope import DishMicroscope


class SimulatedSurvey(Program):
    display_name = "Survey"
    group = "Simulations"
    order = 31
    devices = [SimulatedScanner]
    reads = SURVEY_READS
    inputs = [*REGION_INPUTS, *TILE_INPUTS, *SURVEY_FOCUS_INPUTS]
    cards = {"Region": REGION_INPUTS, "Tiles": TILE_INPUTS, "Focus Search": SURVEY_FOCUS_INPUTS}
    emits = SURVEY_EMITS
    runners = [Single()]
    # Each step of the pipeline starts in a folder holding only the steps before it.
    checks = [CleanFolder()]

    def run(self, ctx: RunContext) -> None:
        # The simulated scanner has no mirrors to turn around, so no overscan.
        run_survey(ctx, DishMicroscope(ctx), (0, 0))
