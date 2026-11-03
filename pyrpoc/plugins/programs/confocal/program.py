"""Confocal: raster the galvo, read the analog inputs, publish a frame."""

from __future__ import annotations

from pyrpoc.core.data import Image
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
)
from ..building_blocks.runners import Continuous, Single


@program_registry.register("confocal")
class Confocal(Program):
    display_name = "Standard"
    group = "Confocal Fluorescence"
    order = 10
    devices = [Galvo, DAQ]
    sections = [SCAN_SECTION, FRAME_COUNT_SECTION, DAQ_SECTION, MODULATION_SECTION]
    emits = {"intensity": Image}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        raster = Raster.for_run(ctx)
        for _ in count_off(ctx, ctx.inputs[NUM_FRAMES], "frame"):
            ctx.publish("intensity", Image(raster.frame(), raster.channel_labels))
