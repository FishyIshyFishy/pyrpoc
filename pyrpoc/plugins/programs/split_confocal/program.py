"""Split confocal: confocal's scan, each pixel's samples split into two
windows, the mask TTL gated to the first, and the raw samples kept too."""

from __future__ import annotations

from dataclasses import replace

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices import DAQ, Galvo
from pyrpoc.plugins.program_io.inputs import (
    MASK_LINES,
    MASKS,
    NUM_FRAMES,
    SAMPLE_RATE_MHZ,
    SCAN_GEOMETRY_INPUTS,
    T0_SAMPLES,
    T1_SAMPLES,
    SplitWindows,
)
from pyrpoc.plugins.program_io.outputs import ImageOutput, SamplesOutput

from ..building_blocks.functions.galvo_raster import Raster
from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.runners import Continuous, Single
from .windows import gated_to_t0, split_frame, split_labels


@program_registry.register("split_confocal")
class SplitConfocal(Program):
    display_name = "Intrapixel split"
    group = "Confocal Fluorescence"
    order = 11
    devices = [Galvo, DAQ]
    sections = [
        InputSection("Scan", SCAN_GEOMETRY_INPUTS),
        InputSection("Frames", (NUM_FRAMES,)),
        InputSection("DAQ", (SAMPLE_RATE_MHZ,)),
        InputSection("Split", (T0_SAMPLES, T1_SAMPLES)),
        InputSection("Modulation", (MASKS, MASK_LINES)),
    ]
    emits = {"intensity": ImageOutput, "raw_pixel_stream": SamplesOutput}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        split = SplitWindows.from_inputs(ctx.inputs)
        raster = Raster.for_run(ctx)
        raster = replace(
            raster, ttl=gated_to_t0(raster.ttl, raster.samples_per_pixel, split.t0_samples)
        )
        labels = split_labels(raster.channel_labels)

        for _ in count_off(ctx, ctx.inputs[NUM_FRAMES], "frame"):
            samples = raster.samples()
            ctx.output("intensity", ImageOutput).add(split_frame(samples, split), labels)
            ctx.output("raw_pixel_stream", SamplesOutput).add(samples, raster.channel_labels)
