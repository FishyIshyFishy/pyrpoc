"""Split confocal: confocal's scan, each pixel's samples split into two
windows, the mask TTL gated to the first, and the raw samples published too."""

from __future__ import annotations

from dataclasses import replace

from pyrpoc.plugins.devices import DAQ, Galvo
from pyrpoc.structs.data_library.data import Image, RawSamples
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.functions.galvo_raster import Raster
from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.parameter_groups import (
    DaqGroup,
    FrameCountGroup,
    ModulationGroup,
    ScanGroup,
    SplitGroup,
)
from ..building_blocks.runners import Continuous, Single
from .windows import gated_to_t0, split_frame, split_labels


@program_registry.register("split_confocal")
class SplitConfocal(Program):
    display_name = "Intrapixel split"
    group = "Confocal Fluorescence"
    order = 11
    uses = [Galvo, DAQ]
    params = [ScanGroup, FrameCountGroup, DaqGroup, SplitGroup, ModulationGroup]
    emits = {"intensity": Image, "raw_pixel_stream": RawSamples}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        split = ctx.params[SplitGroup]
        raster = Raster.for_run(ctx)
        raster = replace(
            raster, ttl=gated_to_t0(raster.ttl, raster.samples_per_pixel, split.t0_samples)
        )
        labels = split_labels(raster.channel_labels)

        for _ in count_off(ctx, ctx.params[FrameCountGroup].num_frames, "frame"):
            samples = raster.samples()
            ctx.publish("intensity", Image(split_frame(samples, split), labels))
            ctx.publish("raw_pixel_stream", RawSamples(samples))
