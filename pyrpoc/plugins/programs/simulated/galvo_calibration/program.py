"""Simulated Galvo Calibration: the calibration, measured on the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices.simulated import SimulatedScanner
from pyrpoc.plugins.program_io.inputs import SCAN_GEOMETRY_INPUTS, ScanGeometry
from pyrpoc.plugins.program_io.outputs import GalvoCalibration, Image

from ...building_blocks.calibration import calibrate
from ...building_blocks.runners import Single
from ..dish_microscope import DishMicroscope


class SimulatedGalvoCalibration(Program):
    display_name = "Galvo Calibration"
    group = "Simulations"
    order = 30
    devices = [SimulatedScanner]
    sections = [InputSection("Scan", SCAN_GEOMETRY_INPUTS)]
    emits = {"calibration": GalvoCalibration, "view": Image}
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        calibrate(ctx, DishMicroscope(ctx), ScanGeometry.from_inputs(ctx.inputs))
