"""Galvo Calibration: µm per volt on each axis, measured with the Prior stage as the ruler."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.plugins.program_io.inputs import SAMPLE_RATE_MHZ, SCAN_GEOMETRY_INPUTS, ScanGeometry
from pyrpoc.plugins.program_io.outputs import GalvoCalibration, Image

from ..building_blocks.calibration import calibrate
from ..building_blocks.rig import RigMicroscope
from ..building_blocks.runners import Single


@program_registry.register("galvo_calibration")
class GalvoCalibrationProgram(Program):
    display_name = "Galvo Calibration"
    group = "Autonomous"
    order = 30
    devices = [Galvo, DAQ, PriorStage]
    sections = [
        InputSection("Scan", SCAN_GEOMETRY_INPUTS),
        InputSection("DAQ", (SAMPLE_RATE_MHZ,)),
    ]
    emits = {"calibration": GalvoCalibration, "view": Image}
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        calibrate(ctx, RigMicroscope(ctx), ScanGeometry.from_inputs(ctx.inputs))
