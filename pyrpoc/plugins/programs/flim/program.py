"""FLIM: scan the galvo emitting tagger markers, read back a histogram cube.

The tagger is set up once per run, outside the frame loop, and torn down in a
``finally`` that also runs on a stop.
"""

from __future__ import annotations

import numpy as np

from pyrpoc.plugins.devices import DAQ, FlimMeasurement, Galvo, TimeTagger
from pyrpoc.structs.data_library.data import Cube3D, Image2D
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.parameter_groups import (
    DaqGroup,
    FrameCountGroup,
    HistogramGroup,
    ScanGroup,
    TriggerGroup,
)
from ..building_blocks.runners import Continuous, Single
from .scan import flim_scan


def read_histograms(flim: FlimMeasurement, n_bins: int, scan: ScanGroup) -> np.ndarray:
    """The just-scanned frame as a ``(H, W, n_bins)`` float32 cube, with the
    overscan columns dropped."""
    histograms = np.asarray(flim.getCurrentFrameEx().getHistograms(), dtype=np.float32)
    return histograms.reshape(scan.y_pixels, scan.total_x, n_bins)[:, scan.kept_columns, :]


def acquire_frame(ctx: RunContext, flim: FlimMeasurement) -> None:
    scan = ctx.params[ScanGroup]
    histogram = ctx.params[HistogramGroup]
    flim_scan(
        ctx.devices[DAQ],
        ctx.devices[Galvo],
        scan,
        ctx.params[DaqGroup].sample_rate_hz,
        ctx.params[TriggerGroup],
    )
    ctx.sleep(histogram.frame_settle_s)
    cube = read_histograms(flim, histogram.histogram_bins, scan)
    ctx.publish("histogram", cube)
    # Photon counts summed over the decay: the intensity image.
    ctx.publish("intensity", cube.sum(axis=2)[np.newaxis], channels=["intensity"])


@program_registry.register("flim")
class FLIM(Program):
    display_name = "FLIM"
    group = "FLIM"
    order = 40
    uses = [Galvo, DAQ, TimeTagger]
    params = [ScanGroup, FrameCountGroup, DaqGroup, TriggerGroup, HistogramGroup]
    emits = {"intensity": Image2D, "histogram": Cube3D}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        scan = ctx.params[ScanGroup]
        histogram = ctx.params[HistogramGroup]
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
            for _ in count_off(ctx, ctx.params[FrameCountGroup].num_frames, "frame"):
                acquire_frame(ctx, flim)
        finally:
            tagger.stop_flim_measurement(flim)
