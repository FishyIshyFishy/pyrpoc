"""Auto-segment: confocal, plus an Otsu mask of each frame's first channel."""

from __future__ import annotations

from pyrpoc.plugins.devices import DAQ, Galvo
from pyrpoc.structs.data_library.data import Image2D, Mask2D
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.functions.galvo_raster import Raster
from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.parameter_groups import DaqGroup, FrameCountGroup, ModulationGroup, ScanGroup
from ..building_blocks.runners import Continuous, Single
from .otsu import otsu_mask


@program_registry.register("auto_segment")
class AutoSegment(Program):
    display_name = "Auto Segment"
    group = "Confocal Fluorescence"
    order = 14
    uses = [Galvo, DAQ]
    params = [ScanGroup, FrameCountGroup, DaqGroup, ModulationGroup]
    emits = {"intensity": Image2D, "mask": Mask2D}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        raster = Raster.for_run(ctx)
        ctx.describe("mask", method="otsu", channel=raster.channel_labels[0])
        for _ in count_off(ctx, ctx.params[FrameCountGroup].num_frames, "frame"):
            frame = raster.frame()
            ctx.publish("intensity", frame, channels=raster.channel_labels)
            ctx.publish("mask", otsu_mask(frame[0]))
