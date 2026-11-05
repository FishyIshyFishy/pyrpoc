"""Galvo Calibration: µm per volt on each axis, measured with the Prior stage as the ruler."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.plugins.program_io.inputs import SAMPLE_RATE_MHZ, SCAN_GEOMETRY_INPUTS, ScanGeometry

from ..building_blocks.multisite import (
    CALIBRATION_EMITS,
    CleanFolder,
    RigMicroscope,
    run_calibration,
)
from ..building_blocks.runners import Single


@program_registry.register("galvo_calibration")
class GalvoCalibrationProgram(Program):
    display_name = "Galvo Calibration"
    group = "Autonomous"
    order = 30
    devices = [Galvo, DAQ, PriorStage]
    inputs = [*SCAN_GEOMETRY_INPUTS, SAMPLE_RATE_MHZ]
    cards = {"Scan": SCAN_GEOMETRY_INPUTS, "DAQ": (SAMPLE_RATE_MHZ,)}
    emits = CALIBRATION_EMITS
    runners = [Single()]
    # Each step of the pipeline starts in a folder holding only the steps before it.
    checks = [CleanFolder(carries_on=False)]

    def run(self, ctx: RunContext) -> None:
        run_calibration(ctx, RigMicroscope(ctx), ScanGeometry.from_inputs(ctx.inputs))
