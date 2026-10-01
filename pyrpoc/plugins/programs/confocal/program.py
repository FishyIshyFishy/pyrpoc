"""Confocal: raster the galvo, read the analog inputs, publish a frame."""

from __future__ import annotations

from pyrpoc.plugins.devices import DAQ, Galvo
from pyrpoc.structs.data_library.data import Image2D
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.functions.galvo_raster import Raster
from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.parameter_groups import DaqGroup, FrameCountGroup, ModulationGroup, ScanGroup
from ..building_blocks.runners import Continuous, Single


@program_registry.register("confocal")
class Confocal(Program):
    display_name = "Standard"
    group = "Confocal Fluorescence"
    order = 10
    uses = [Galvo, DAQ]
    params = [ScanGroup, FrameCountGroup, DaqGroup, ModulationGroup]
    emits = {"intensity": Image2D}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        raster = Raster.for_run(ctx)
        for _ in count_off(ctx, ctx.params[FrameCountGroup].num_frames, "frame"):
            ctx.publish("intensity", raster.frame(), channels=raster.channel_labels)
