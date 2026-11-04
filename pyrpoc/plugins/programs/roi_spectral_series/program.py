"""ROI spectral series: sweep the galvos over one masked region while the CCD reads spectra
back to back, to follow how the spectrum shifts as the dye bleaches."""

# Each spectrum averages the whole region. The sweep starts before the first read and stops after
# the last, so the region is lit through the camera's readout too. Each read's start and end, in
# seconds after the sweep began, are recorded beside the sweep's period, so how many sweeps a
# spectrum spans can be worked out afterwards. They are host times around each read, not the
# camera's.

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices import DAQ, AndorCCD, Galvo
from pyrpoc.plugins.program_io.inputs import (
    CCD_EXPOSURE_S,
    CCD_FRAMES_TO_AVERAGE,
    CENTER_REGION_FRACTION,
    MASKS,
    NUM_FRAMES,
    SAMPLE_RATE_MHZ,
    SCAN_GEOMETRY_INPUTS,
    Mask,
    ScanGeometry,
    rate_hz,
)
from pyrpoc.plugins.program_io.outputs import Spectrum
from pyrpoc.plugins.programs.building_blocks.functions.ccd_read import spectrum_readout

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
from ..building_blocks.runners import Single


def plan_sweep(
    scan: ScanGeometry, masks: tuple[Mask, ...], center_fraction: float, sample_rate_hz: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """The region's looping waveform, and what the recording says of it. The
    dwell is the one played, which truncates to whole samples."""
    samples_per_pixel = pixel_samples(scan.dwell_time_us, sample_rate_hz)
    spans = row_spans(roi_region(masks, center_fraction, scan))
    waveform = roi_waveform(scan, spans, samples_per_pixel)
    pixels_swept = waveform.shape[1] // samples_per_pixel
    metadata = region_layout(region_source(masks), spans, Box.around(spans), pixels_swept)
    metadata["dwell_s"] = samples_per_pixel / sample_rate_hz
    metadata["period_s"] = waveform.shape[1] / sample_rate_hz
    return waveform, metadata


@program_registry.register("roi_spectral_series")
class RoiSpectralSeries(Program):
    display_name = "ROI sweep + spectral series"
    group = "Custom"
    order = 50
    devices = [Galvo, AndorCCD]
    sections = [
        InputSection("Scan", SCAN_GEOMETRY_INPUTS),
        InputSection("DAQ", (SAMPLE_RATE_MHZ,)),
        InputSection("Region", (MASKS, CENTER_REGION_FRACTION)),
        InputSection("Frames", (NUM_FRAMES,)),
        InputSection("CCD Readout", (CCD_EXPOSURE_S, CCD_FRAMES_TO_AVERAGE)),
    ]
    emits = {"spectrum": Spectrum}
    # Once only: each start is one bleaching series of a set length.
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        sample_rate_hz = rate_hz(ctx.inputs)
        waveform, sweep = plan_sweep(
            ScanGeometry.from_inputs(ctx.inputs),
            ctx.inputs[MASKS],
            ctx.inputs[CENTER_REGION_FRACTION],
            sample_rate_hz,
        )
        ctx.describe("spectrum", sweep=sweep)
        ccd = ctx.devices[AndorCCD]
        readout = spectrum_readout(ctx.inputs)
        # Before the sweep, so the first read starts as soon as the region is swept.
        configure_spectrum(ctx, ccd, readout, "spectrum")
        times: list[list[float]] = []
        # The DAQ is bound though ``devices`` does not name it: claims follow ``backed_by``.
        with sweeping(ctx.devices[DAQ], ctx.devices[Galvo], waveform, sample_rate_hz) as began:
            for _ in count_off(ctx, ctx.inputs[NUM_FRAMES], "spectrum"):
                start = time.monotonic() - began
                read_spectrum(ctx, ccd, readout, "spectrum")
                times.append([start, time.monotonic() - began])
                # Every frame, so a run cut short still says when its spectra were taken.
                ctx.describe("spectrum", frame_times_s=times)
