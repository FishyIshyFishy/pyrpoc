"""Raman spectrum: read a spectrum off the Andor CCD wherever the beam already is.

Run continuously, it is the live view used to align and focus. The x axis is
pixel index; wavelength calibration is not part of the recording.
"""

from __future__ import annotations

from pyrpoc.plugins.devices import AndorCCD
from pyrpoc.structs.data_library.data import Spectrum1D
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.functions.ccd_read import publish_spectrum
from ..building_blocks.parameter_groups import CcdReadoutGroup
from ..building_blocks.runners import Continuous, Single


@program_registry.register("raman")
class Raman(Program):
    display_name = "Spectral readout"
    group = "Raman"
    order = 10
    uses = [AndorCCD]
    params = [CcdReadoutGroup]
    emits = {"spectrum": Spectrum1D}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        readout = ctx.params[CcdReadoutGroup].to_readout()
        publish_spectrum(ctx, ctx.devices[AndorCCD], readout, "spectrum")
