"""One averaged spectrum off the Andor CCD, published from inside a run."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from pyrpoc.plugins.devices import AndorCCD, SpectrumReadout
from pyrpoc.plugins.devices.andor_ccd.device import clipped
from pyrpoc.structs.plugins.programs.program import Cancelled, RunContext


def publish_spectrum(ctx: RunContext, ccd: AndorCCD, readout: SpectrumReadout, output: str) -> None:
    """Configure the camera, take one spectrum and publish it to ``output`` as
    counts per frame. The camera sums its accumulations, so the mean is the sum
    over their number; clipping is judged on the sum, where the ADC ceiling is."""
    timings = ccd.configure(readout)
    ctx.describe(output, camera=ccd.metadata(), readout=asdict(readout), accepted=asdict(timings))
    ctx.status(f"reading {readout.accumulations} x {timings.exposure_s:g} s")
    try:
        counts = ccd.acquire(ctx.sleep)
    except Cancelled:
        ccd.abort()
        raise
    averaged = counts.astype(np.float32) / readout.accumulations
    ctx.publish(output, averaged[None, :], channels=["ccd"])
    if clipped(counts, readout.accumulations):
        ctx.status("spectrum saturated")
