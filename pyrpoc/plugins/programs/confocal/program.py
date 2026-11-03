"""Confocal: raster the galvo, read the analog inputs, publish a frame."""

from __future__ import annotations

from pyrpoc.core.data import Image
from pyrpoc.core.input_section import InputSection
from pyrpoc.core.mask import MASKS
from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.plugins.devices import DAQ, Galvo

from ..building_blocks.functions.galvo_raster import Raster
from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.program_inputs import (
    MASK_LINES,
    NUM_FRAMES,
    SAMPLE_RATE_HZ,
    SCAN_GEOMETRY_INPUTS,
)
from ..building_blocks.runners import Continuous, Single


@program_registry.register("confocal")
class Confocal(Program):
    display_name = "Standard"
    group = "Confocal Fluorescence"
    order = 10
    devices = [Galvo, DAQ]
    sections = [
        InputSection("Scan", SCAN_GEOMETRY_INPUTS),
        InputSection("Frames", (NUM_FRAMES,)),
        InputSection("DAQ", (SAMPLE_RATE_HZ,)),
        InputSection("Modulation", (MASKS, MASK_LINES)),
    ]
    emits = {"intensity": Image}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        raster = Raster.for_run(ctx)
        for _ in count_off(ctx, ctx.inputs[NUM_FRAMES], "frame"):
            ctx.publish("intensity", Image(raster.frame(), raster.channel_labels))
