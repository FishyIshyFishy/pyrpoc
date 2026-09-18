"""Spectrum: read the CCD, publish the trace. No beam, no scanning.

The simplest program in the folder, and useful on its own: it is how you check
alignment, take a dark frame, set an exposure and see whether the sample
fluoresces before anyone picks a point. Pinpoint Raman is this program with the
galvos parked first, which is why the two share ``ReadoutGroup``.

Setup is outside the loop, the same shape ``flim.py`` uses around the
TimeTagger. The difference is what the ``finally`` does: the camera is returned
to idle rather than shut down, because the handle is meant to outlive the run.
Closing it here would cost seconds on the next run and throw away the cooling.

``pinpoint_raman.py`` holds a copy of the readout loop below rather than
importing it, the way ``split_confocal.py`` holds its copy of the waveform
arithmetic. The copies are meant to stay identical: change one and change the
other, or say in the docstring why they now differ.

The spectral axis is the pixel index. Turning pixels into wavelengths is a
calibration done outside this application.
"""

from __future__ import annotations

import numpy as np

from pyrpoc.core.errors import CcdError
from pyrpoc.core.streams import Spectrum1D
from pyrpoc.devices import CCD
from pyrpoc.run.program import Program

from .components import ReadoutGroup
from .registry import program_registry


@program_registry.register("ccd_spectrum")
class Spectrum(Program):
    uses = [CCD]
    params = [ReadoutGroup]
    emits = {"spectrum": Spectrum1D}

    def run(self, ctx) -> None:
        readout = ctx.params[ReadoutGroup]
        ccd: CCD = ctx.devices[CCD]

        ctx.status("opening the camera")
        ccd.open()
        try:
            ccd.configure_for_spectrum(exposure_s=readout.exposure_s)

            # Read-backs, once, before the first spectrum. What the camera was
            # *configured* with is already written into the run metadata by the
            # runner, from the device config; what it settled on is not, and
            # the two differ -- the exposure especially. Everything here is
            # JSON-native, because the metadata file serialises with
            # ``default=str`` and would turn an array into its own elided repr.
            ccd.temperature()
            ctx.describe("spectrum", **ccd.instrument_metadata())

            for _ in ctx.frames(1):
                spectrum = read_averaged(ctx, ccd, readout.num_acquisitions)
                ctx.publish("spectrum", spectrum, channels=["raman"])
        finally:
            # Idle, not closed. The handle outlives the run on purpose.
            ccd.abort()


def read_averaged(ctx, ccd: CCD, num_acquisitions: int) -> np.ndarray:
    """Read ``num_acquisitions`` spectra off the camera and average them.

    A copy of the loop in ``pinpoint_raman.py``, kept identical there for the
    same reason the rest of the readout loop is: change one and change the
    other, or say in the docstring why they now differ.
    """
    total = max(int(num_acquisitions), 1)
    accumulated: np.ndarray | None = None
    for shot in range(total):
        if total > 1:
            ctx.status(f"acquisition {shot + 1}/{total}")
        else:
            ctx.status("spectrum")

        spectrum = ccd.read_spectrum(should_stop=ctx.cancelled)
        if spectrum is None:
            # Stopped mid-exposure. ``check_cancel`` turns that into the
            # Cancelled the runner treats as a clean stop; the raise is for
            # the case that cannot happen, so that it cannot happen silently.
            ctx.check_cancel()
            raise CcdError("the camera returned no data and was not stopped")

        if ccd.saturated:
            ctx.status(f"acquisition {shot + 1}/{total} - saturated at {ccd.last_max_counts} counts")
        accumulated = spectrum if accumulated is None else accumulated + spectrum

    assert accumulated is not None
    return (accumulated / total).astype(np.float32, copy=False)
