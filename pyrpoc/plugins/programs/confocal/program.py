"""Confocal: raster the galvo, read the analog inputs, add a frame."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.plugins.devices import DAQ, Galvo
from pyrpoc.plugins.program_io.inputs import (
    MASK_LINES,
    MASKS,
    NUM_FRAMES,
    SAMPLE_RATE_MHZ,
    SCAN_GEOMETRY_INPUTS,
)
from pyrpoc.plugins.program_io.outputs import ImageOutput

from ..building_blocks.functions.galvo_raster import Raster
from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.runners import Continuous, Single


@program_registry.register("confocal")
class Confocal(Program):
    display_name = "Standard"
    group = "Confocal Fluorescence"
    order = 10
    devices = [Galvo, DAQ]
    inputs = [*SCAN_GEOMETRY_INPUTS, NUM_FRAMES, SAMPLE_RATE_MHZ, MASKS, MASK_LINES]
    # MASKS has no row: masks are drawn and edited in a Mask Editor.
    cards = {
        "Scan": SCAN_GEOMETRY_INPUTS,
        "Frames": (NUM_FRAMES,),
        "DAQ": (SAMPLE_RATE_MHZ,),
        "Modulation": (MASK_LINES,),
    }
    emits = {"intensity": ImageOutput}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        raster = Raster.for_run(ctx)
        for _ in count_off(ctx, ctx.inputs[NUM_FRAMES], "frame"):
            ctx.output("intensity", ImageOutput).add(raster.frame(), raster.channel_labels)
