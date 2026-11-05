"""Averaged spectra off the Andor CCD, added to a spectrum output from inside a run. A series
configures the camera once and then reads it frame after frame."""

from __future__ import annotations

import numpy as np

from pyrpoc.core.program import Cancelled, RunContext
from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.plugins.devices import AndorCCD, SpectrumReadout
from pyrpoc.plugins.devices.andor_ccd.device import AcceptedTimings, Amplifier, ReadMode, clipped
from pyrpoc.plugins.program_io.inputs import CCD_EXPOSURE_S, CCD_FRAMES_TO_AVERAGE
from pyrpoc.plugins.program_io.outputs import SpectrumOutput

# The rest of the readout is fixed at what a Raman spectrum wants: every row
# binned, read slowly through the conventional amplifier for the least noise.
# A track is unused under full vertical binning.
RAMAN_READ_MODE: ReadMode = "fvb"
RAMAN_AMPLIFIER: Amplifier = "conventional"
RAMAN_READOUT_MHZ = 0.05
RAMAN_PREAMP_GAIN = 1.0


def spectrum_readout(values: ProgramInputValues) -> SpectrumReadout:
    """The readout a Raman spectrum is taken with."""
    return SpectrumReadout(
        exposure_s=values[CCD_EXPOSURE_S],
        accumulations=values[CCD_FRAMES_TO_AVERAGE],
        read_mode=RAMAN_READ_MODE,
        track_centre=1,
        track_height=1,
        amplifier=RAMAN_AMPLIFIER,
        em_gain=1,
        readout_mhz=RAMAN_READOUT_MHZ,
        preamp_gain=RAMAN_PREAMP_GAIN,
    )


def configure_spectrum(
    ctx: RunContext, ccd: AndorCCD, readout: SpectrumReadout, output: str
) -> AcceptedTimings:
    """Program the camera for ``readout`` and record the exposure it accepted on ``output``."""
    timings = ccd.configure(readout)
    ctx.output(output, SpectrumOutput).set_exposure(timings.exposure_s)
    return timings


def read_spectrum(ctx: RunContext, ccd: AndorCCD, readout: SpectrumReadout, output: str) -> None:
    """Take one spectrum as configured and add it to ``output`` as counts per frame."""
    # The camera sums its accumulations, so the mean is the sum over their number; clipping is
    # judged on the sum, where the ADC ceiling is.
    try:
        counts = ccd.acquire(ctx.sleep)
    except Cancelled:
        ccd.abort()
        raise
    averaged = counts.astype(np.float32) / readout.accumulations
    ctx.output(output, SpectrumOutput).add(averaged[None, :], ["ccd"])
    if clipped(counts, readout.accumulations):
        ctx.status("spectrum saturated")


def take_spectrum(ctx: RunContext, ccd: AndorCCD, readout: SpectrumReadout, output: str) -> None:
    """Configure the camera, take one spectrum and add it to ``output``."""
    timings = configure_spectrum(ctx, ccd, readout, output)
    ctx.status(f"reading {readout.accumulations} x {timings.exposure_s:g} s")
    read_spectrum(ctx, ccd, readout, output)
