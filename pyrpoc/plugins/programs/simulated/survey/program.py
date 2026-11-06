"""Simulated Survey: the survey, over the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import RunContext
from pyrpoc.plugins.devices.simulated import SimulatedScanner

from ...building_blocks.multisite import Microscope, SurveyProgram
from ..dish_microscope import DishMicroscope


class SimulatedSurvey(SurveyProgram):
    display_name = "Survey"
    group = "Simulations"
    order = 31
    devices = [SimulatedScanner]

    def microscope(self, ctx: RunContext) -> Microscope:
        return DishMicroscope(ctx)

    def overscan(self, ctx: RunContext) -> tuple[int, int]:
        # The simulated scanner has no mirrors to turn around.
        return (0, 0)
