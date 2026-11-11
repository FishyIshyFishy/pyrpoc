"""Simulated Galvo Calibration: the calibration, measured on the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import RunContext, program_registry
from pyrpoc.plugins.devices.simulated import SimulatedScanner

from ...building_blocks.multisite import CalibrationProgram, Microscope
from ..dish_microscope import DishMicroscope


@program_registry.register("simulated_galvo_calibration")
class SimulatedGalvoCalibration(CalibrationProgram):
    display_name = "Galvo Calibration"
    group = "Simulations"
    order = 30
    devices = [SimulatedScanner]

    def microscope(self, ctx: RunContext) -> Microscope:
        return DishMicroscope(ctx)
