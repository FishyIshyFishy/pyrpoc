"""Simulated Galvo Calibration: the calibration, measured on the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext
from pyrpoc.plugins.devices.simulated import SimulatedScanner
from pyrpoc.plugins.program_io.inputs import SCAN_GEOMETRY_INPUTS, ScanGeometry

from ...building_blocks.multisite import CALIBRATION_EMITS, CleanFolder, run_calibration
from ...building_blocks.runners import Single
from ..dish_microscope import DishMicroscope


class SimulatedGalvoCalibration(Program):
    display_name = "Galvo Calibration"
    group = "Simulations"
    order = 30
    devices = [SimulatedScanner]
    inputs = list(SCAN_GEOMETRY_INPUTS)
    cards = {"Scan": SCAN_GEOMETRY_INPUTS}
    emits = CALIBRATION_EMITS
    runners = [Single()]
    # Each step of the pipeline starts in a folder holding only the steps before it.
    checks = [CleanFolder(carries_on=False)]

    def run(self, ctx: RunContext) -> None:
        run_calibration(ctx, DishMicroscope(ctx), ScanGeometry.from_inputs(ctx.inputs))
