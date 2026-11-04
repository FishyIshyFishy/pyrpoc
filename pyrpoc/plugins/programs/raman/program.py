"""Raman spectrum: read a spectrum off the Andor CCD wherever the beam already is."""

# Run continuously, it is the live view used to align and focus. The x axis is pixel index;
# wavelength calibration is not part of the recording.

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices import AndorCCD
from pyrpoc.plugins.program_io.inputs import CCD_EXPOSURE_S, CCD_FRAMES_TO_AVERAGE
from pyrpoc.plugins.program_io.outputs import Spectrum
from pyrpoc.plugins.programs.building_blocks.functions.ccd_read import spectrum_readout

from ..building_blocks.functions.ccd_read import publish_spectrum
from ..building_blocks.runners import Continuous, Single


@program_registry.register("raman")
class Raman(Program):
    display_name = "Spectral readout"
    group = "Raman"
    order = 10
    devices = [AndorCCD]
    sections = [
        InputSection("CCD Readout", (CCD_EXPOSURE_S, CCD_FRAMES_TO_AVERAGE)),
    ]
    emits = {"spectrum": Spectrum}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        readout = spectrum_readout(ctx.inputs)
        publish_spectrum(ctx, ctx.devices[AndorCCD], readout, "spectrum")
