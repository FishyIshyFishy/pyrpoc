"""Raman spectrum: read a spectrum off the Andor CCD wherever the beam already is.

Run continuously, it is the live view used to align and focus. The x axis is
pixel index; wavelength calibration is not part of the recording.
"""

from __future__ import annotations

from pyrpoc.core.data import Spectrum
from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.plugins.devices import AndorCCD

from ..building_blocks.functions.ccd_read import publish_spectrum
from ..building_blocks.program_inputs import CCD_READOUT_SECTION, spectrum_readout
from ..building_blocks.runners import Continuous, Single


@program_registry.register("raman")
class Raman(Program):
    display_name = "Spectral readout"
    group = "Raman"
    order = 10
    devices = [AndorCCD]
    sections = [CCD_READOUT_SECTION]
    emits = {"spectrum": Spectrum}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        readout = spectrum_readout(ctx.inputs)
        publish_spectrum(ctx, ctx.devices[AndorCCD], readout, "spectrum")
