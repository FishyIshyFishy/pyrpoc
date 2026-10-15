"""CCD spectrum: read spectra off the Andor CCD wherever the beam already is.

Run continuously, it is the live view used to align and focus. The x axis is
pixel index; wavelength calibration is not part of the recording.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from pyrpoc.plugins.devices import AndorCCD, AndorError, SpectrumReadout
from pyrpoc.plugins.devices.andor_ccd.device import clipped
from pyrpoc.structs.data_library.data import Spectrum1D
from pyrpoc.structs.plugins.programs.program import (
    Cancelled,
    Program,
    RunContext,
    program_registry,
)

from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.parameter_groups import CcdReadoutGroup, FrameCountGroup
from ..building_blocks.runners import Continuous, Single


def check_em_safe(readout: SpectrumReadout) -> None:
    """Saturating through the EM register ages the sensor, so the run stops
    rather than repeating it; on the conventional amplifier it is only clipping."""
    if readout.amplifier == "em":
        raise AndorError(
            f"spectrum saturated on the EM amplifier at gain {readout.em_gain}; stopped to "
            "protect the sensor. Lower the gain or exposure, or use the conventional amplifier"
        )


@program_registry.register("ccd_spectrum")
class CcdSpectrum(Program):
    display_name = "CCD spectrum"
    group = "Spectrum"
    order = 10
    uses = [AndorCCD]
    params = [CcdReadoutGroup, FrameCountGroup]
    emits = {"spectrum": Spectrum1D}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        ccd = ctx.devices[AndorCCD]
        readout = ctx.params[CcdReadoutGroup].to_readout()
        frames = ctx.params[FrameCountGroup].num_frames

        timings = ccd.configure(readout)
        ctx.describe(
            "spectrum", camera=ccd.metadata(), readout=asdict(readout), accepted=asdict(timings)
        )
        try:
            for index in count_off(ctx, frames, "spectrum"):
                counts = ccd.acquire(ctx.sleep)
                ctx.publish("spectrum", counts[None, :].astype(np.float32), channels=["ccd"])
                if clipped(counts, readout.accumulations):
                    check_em_safe(readout)
                    ctx.status(f"spectrum {index + 1}/{frames} saturated")
        except Cancelled:
            ccd.abort()
            raise
