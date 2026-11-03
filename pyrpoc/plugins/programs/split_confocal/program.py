"""Split confocal: confocal's scan, each pixel's samples split into two
windows, the mask TTL gated to the first, and the raw samples published too."""

from __future__ import annotations

from dataclasses import replace

from pyrpoc.core.data import Image, RawSamples
from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.plugins.devices import DAQ, Galvo

from ..building_blocks.functions.galvo_raster import Raster
from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.program_inputs import (
    DAQ_SECTION,
    FRAME_COUNT_SECTION,
    MODULATION_SECTION,
    NUM_FRAMES,
    SCAN_SECTION,
    SPLIT_SECTION,
    SplitWindows,
)
from ..building_blocks.runners import Continuous, Single
from .windows import gated_to_t0, split_frame, split_labels


@program_registry.register("split_confocal")
class SplitConfocal(Program):
    display_name = "Intrapixel split"
    group = "Confocal Fluorescence"
    order = 11
    devices = [Galvo, DAQ]
    sections = [SCAN_SECTION, FRAME_COUNT_SECTION, DAQ_SECTION, SPLIT_SECTION, MODULATION_SECTION]
    emits = {"intensity": Image, "raw_pixel_stream": RawSamples}
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
            ctx.publish("intensity", Image(split_frame(samples, split), labels))
            ctx.publish("raw_pixel_stream", RawSamples(samples))
