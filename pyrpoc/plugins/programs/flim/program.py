"""FLIM: scan the galvo emitting tagger markers, read back a histogram cube.

The tagger is set up once per run, outside the frame loop, and torn down in a
``finally`` that also runs on a stop.
"""

from __future__ import annotations

import numpy as np

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices import DAQ, FlimMeasurement, Galvo, TimeTagger
from pyrpoc.plugins.program_io.inputs import (
    FRAME_SETTLE_S,
    FRAME_TRIGGER_PFI,
    HISTOGRAM_BINS,
    HISTOGRAM_BINWIDTH_PS,
    LASER_FREQUENCY_MHZ,
    NUM_FRAMES,
    PIXEL_CLOCK_CTR,
    PIXEL_CLOCK_PFI,
    SAMPLE_RATE_HZ,
    SCAN_GEOMETRY_INPUTS,
    HistogramSettings,
    ScanGeometry,
    TriggerLines,
)
from pyrpoc.plugins.program_io.outputs import DecayHistogram, Image

from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.runners import Continuous, Single
from .scan import flim_scan


def read_histograms(flim: FlimMeasurement, n_bins: int, scan: ScanGeometry) -> np.ndarray:
    """The just-scanned frame as a ``(H, W, n_bins)`` float32 cube, with the
    overscan columns dropped."""
    histograms = np.asarray(flim.getCurrentFrameEx().getHistograms(), dtype=np.float32)
    return histograms.reshape(scan.y_pixels, scan.total_x, n_bins)[:, scan.kept_columns, :]


def acquire_frame(ctx: RunContext, flim: FlimMeasurement) -> None:
    scan = ScanGeometry.from_inputs(ctx.inputs)
    histogram = HistogramSettings.from_inputs(ctx.inputs)
    flim_scan(
        ctx.devices[DAQ],
        ctx.devices[Galvo],
        scan,
        ctx.inputs[SAMPLE_RATE_HZ],
        TriggerLines.from_inputs(ctx.inputs),
    )
    ctx.sleep(histogram.frame_settle_s)
    cube = read_histograms(flim, histogram.histogram_bins, scan)
    ctx.publish("histogram", DecayHistogram(cube))
    # Photon counts summed over the decay: the intensity image.
    ctx.publish("intensity", Image(cube.sum(axis=2)[np.newaxis], ["intensity"]))


@program_registry.register("flim")
class FLIM(Program):
    display_name = "FLIM"
    group = "FLIM"
    order = 40
    devices = [Galvo, DAQ, TimeTagger]
    sections = [
        InputSection("Scan", SCAN_GEOMETRY_INPUTS),
        InputSection("Frames", (NUM_FRAMES,)),
        InputSection("DAQ", (SAMPLE_RATE_HZ,)),
        InputSection("Triggers", (FRAME_TRIGGER_PFI, PIXEL_CLOCK_CTR, PIXEL_CLOCK_PFI)),
        InputSection(
            "Histogram",
            (LASER_FREQUENCY_MHZ, HISTOGRAM_BINS, HISTOGRAM_BINWIDTH_PS, FRAME_SETTLE_S),
        ),
    ]
    emits = {"intensity": Image, "histogram": DecayHistogram}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        scan = ScanGeometry.from_inputs(ctx.inputs)
        histogram = HistogramSettings.from_inputs(ctx.inputs)
        tagger = ctx.devices[TimeTagger]
        ctx.describe(
            "histogram",
            laser_period_ps=histogram.laser_period_ps,
            binwidth_ps=histogram.histogram_binwidth_ps,
            n_bins=histogram.histogram_bins,
        )

        ctx.status("starting the time tagger")
        tagger.create_tagger()
        tagger.configure_for_flim()
        flim = tagger.start_flim_measurement(
            n_pixels=scan.total_x * scan.y_pixels,
            n_bins=histogram.histogram_bins,
            binwidth_ps=histogram.histogram_binwidth_ps,
        )
        try:
            for _ in count_off(ctx, ctx.inputs[NUM_FRAMES], "frame"):
                acquire_frame(ctx, flim)
        finally:
            tagger.stop_flim_measurement(flim)
