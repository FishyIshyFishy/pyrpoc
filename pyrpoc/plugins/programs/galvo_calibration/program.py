"""Galvo Calibration: µm per volt on each axis, measured with the Prior stage as the ruler."""

from __future__ import annotations

from pyrpoc.core.program import RunContext, program_registry
from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.plugins.program_io.inputs import SAMPLE_RATE_MHZ

from ..building_blocks.multisite import CalibrationProgram, Microscope, RigMicroscope


@program_registry.register("galvo_calibration")
class GalvoCalibrationProgram(CalibrationProgram):
    display_name = "Galvo Calibration"
    group = "Autonomous"
    order = 30
    devices = [Galvo, DAQ, PriorStage]
    inputs = [*CalibrationProgram.inputs, SAMPLE_RATE_MHZ]
    cards = {**CalibrationProgram.cards, "DAQ": (SAMPLE_RATE_MHZ,)}

    def microscope(self, ctx: RunContext) -> Microscope:
        return RigMicroscope(ctx)
