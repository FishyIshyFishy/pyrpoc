"""ROI spectral series: sweep the galvos over one masked region without pause
while the CCD reads spectra back to back, to follow how the spectrum shifts as
the dye bleaches.

Each spectrum averages the whole region. The sweep starts before the first
read and stops after the last, so the region is lit through the camera's
readout too. Each read's start and end, in seconds after the sweep began, are
recorded beside the sweep's period, so how many sweeps a spectrum spans can be
worked out afterwards. They are host times around each read, not the camera's.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pyrpoc.plugins.devices import DAQ, AndorCCD, Galvo
from pyrpoc.structs.data_library.data import Spectrum1D
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.functions.ccd_read import configure_spectrum, read_spectrum
from ..building_blocks.functions.daq_tasks import sweeping
from ..building_blocks.functions.galvo_raster import pixel_samples
from ..building_blocks.functions.roi_scan import (
    Box,
    region_layout,
    region_source,
    roi_region,
    roi_waveform,
    row_spans,
)
from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.parameter_groups import (
    CcdReadoutGroup,
    DaqGroup,
    FrameCountGroup,
    RoiGroup,
    ScanGroup,
)
from ..building_blocks.runners import Single


def plan_sweep(
    scan: ScanGroup, roi: RoiGroup, sample_rate_hz: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """The region's looping waveform, and what the recording says of it. The
    dwell is the one played, which truncates to whole samples."""
    samples_per_pixel = pixel_samples(scan.dwell_time_us, sample_rate_hz)
    spans = row_spans(roi_region(roi, scan))
    waveform = roi_waveform(scan, spans, samples_per_pixel)
    pixels_swept = waveform.shape[1] // samples_per_pixel
    metadata = region_layout(region_source(roi), spans, Box.around(spans), pixels_swept)
    metadata["dwell_s"] = samples_per_pixel / sample_rate_hz
    metadata["period_s"] = waveform.shape[1] / sample_rate_hz
    return waveform, metadata


@program_registry.register("roi_spectral_series")
class RoiSpectralSeries(Program):
    display_name = "ROI sweep + spectral series"
    group = "Custom"
    order = 50
    uses = [Galvo, AndorCCD]
    params = [ScanGroup, DaqGroup, RoiGroup, FrameCountGroup, CcdReadoutGroup]
    emits = {"spectrum": Spectrum1D}
    # Once only: each start is one bleaching series of a set length.
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        sample_rate_hz = ctx.params[DaqGroup].sample_rate_hz
        waveform, sweep = plan_sweep(ctx.params[ScanGroup], ctx.params[RoiGroup], sample_rate_hz)
        ctx.describe("spectrum", sweep=sweep)
        ccd = ctx.devices[AndorCCD]
        readout = ctx.params[CcdReadoutGroup].to_readout()
        # Before the sweep, so the first read starts as soon as the region is swept.
        configure_spectrum(ctx, ccd, readout, "spectrum")
        times: list[list[float]] = []
        # The DAQ is bound though ``uses`` does not name it: claims follow ``backed_by``.
        with sweeping(ctx.devices[DAQ], ctx.devices[Galvo], waveform, sample_rate_hz) as began:
            for _ in count_off(ctx, ctx.params[FrameCountGroup].num_frames, "spectrum"):
                start = time.monotonic() - began
                read_spectrum(ctx, ccd, readout, "spectrum")
                times.append([start, time.monotonic() - began])
                # Every frame, so a run cut short still says when its spectra were taken.
                ctx.describe("spectrum", frame_times_s=times)
