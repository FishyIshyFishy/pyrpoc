"""FLIM: scan the galvo emitting tagger markers, read back a histogram cube.

The clearest demonstration of "the program owns the loop". In v3.0
``FlimModality.acquire_once`` called ``setup_tagger()`` at the top and
``teardown_tagger()`` in a ``finally`` -- **per frame** -- because
``acquire_once`` had to be self-contained and there was nowhere else for per-run
setup to go. A ten-frame run created and freed the TimeTagger ten times.

Here setup is simply outside the loop, and the ``finally`` runs on a stop
because cancellation is an exception raised out through ``run()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.core.params import (
    FlimDaqGroup,
    HistogramGroup,
    ScanGroup,
    TriggerGroup,
    group,
    int_field,
)
from pyrpoc.core.streams import Cube3D, Image2D
from pyrpoc.devices import DAQ, Galvo, TimeTagger
from pyrpoc.operations.tagger import flim_intensity, flim_scan, read_flim_frame
from pyrpoc.run.program import Program

from .registry import program_registry


@dataclass
class FlimParams:
    scan: ScanGroup = group(ScanGroup, "Scan")
    daq: FlimDaqGroup = group(FlimDaqGroup, "DAQ")
    triggers: TriggerGroup = group(TriggerGroup, "Triggers")
    histogram: HistogramGroup = group(HistogramGroup, "Histogram")
    num_frames: int = int_field(
        "Frames", 1, minimum=1, tooltip="Number of frames to capture"
    )


@program_registry.register("flim")
class FLIM(Program):
    uses = [Galvo, DAQ, TimeTagger]
    params = FlimParams
    emits = {"intensity": Image2D, "histogram": Cube3D}

    def run(self, ctx) -> None:
        p: FlimParams = ctx.params
        daq: DAQ = ctx.devices[DAQ]
        galvo: Galvo = ctx.devices[Galvo]
        tagger: TimeTagger = ctx.devices[TimeTagger]

        total_x = p.scan.total_x
        ctx.describe(
            "histogram",
            laser_period_ps=p.histogram.laser_period_ps,
            binwidth_ps=p.histogram.histogram_binwidth_ps,
            n_bins=p.histogram.histogram_bins,
        )

        ctx.status("starting the time tagger")
        tagger.create_tagger()
        tagger.configure_for_flim()
        flim = tagger.start_flim_measurement(
            n_pixels=total_x * p.scan.y_pixels,
            n_bins=p.histogram.histogram_bins,
            binwidth_ps=p.histogram.histogram_binwidth_ps,
        )
        try:
            total = "" if ctx.continuous else f"/{p.num_frames}"
            for index in ctx.frames(p.num_frames):
                ctx.status(f"frame {index + 1}{total}")
                flim_scan(
                    **p.scan,
                    **p.daq,
                    **p.triggers,
                    **daq.config,
                    **galvo.config,
                )
                ctx.sleep(p.histogram.frame_settle_s)
                cube = read_flim_frame(
                    flim,
                    n_bins=p.histogram.histogram_bins,
                    y_pixels=p.scan.y_pixels,
                    total_x_pixels=total_x,
                    extra_left=p.scan.extra_left,
                    x_pixels=p.scan.x_pixels,
                )
                ctx.publish("histogram", cube)
                ctx.publish(
                    "intensity",
                    flim_intensity(cube)[np.newaxis],
                    channels=["intensity"],
                )
        finally:
            tagger.stop_flim_measurement(flim)
