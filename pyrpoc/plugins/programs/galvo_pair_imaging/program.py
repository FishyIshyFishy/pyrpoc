"""Galvo imaging: confocal's scan on the reference galvo, the second, or both on one clock,
always over the same field of the sample."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.plugins.devices import DAQ
from pyrpoc.plugins.program_io.inputs import (
    MASK_LINES,
    MASKS,
    NUM_FRAMES,
    REGISTRATION_INPUTS,
    SAMPLE_RATE_MHZ,
    SCAN_GALVOS,
    SCAN_GEOMETRY_INPUTS,
)
from pyrpoc.plugins.program_io.outputs import ImageOutput

from ..building_blocks.functions.galvo_pair import GALVO_PAIR, scan_drives
from ..building_blocks.functions.galvo_raster import Raster
from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.runners import Continuous, Single


@program_registry.register("galvo_pair_imaging")
class GalvoPairImaging(Program):
    display_name = "Galvo imaging"
    group = "Karsten"
    order = 300
    devices = [DAQ]
    chooses = GALVO_PAIR
    inputs = [
        SCAN_GALVOS,
        *REGISTRATION_INPUTS,
        *SCAN_GEOMETRY_INPUTS,
        NUM_FRAMES,
        SAMPLE_RATE_MHZ,
        MASKS,
        MASK_LINES,
    ]
    # MASKS has no row: masks are drawn and edited in a Mask Editor.
    cards = {
        "Galvos": (SCAN_GALVOS, *REGISTRATION_INPUTS),
        "Scan": SCAN_GEOMETRY_INPUTS,
        "Frames": (NUM_FRAMES,),
        "DAQ": (SAMPLE_RATE_MHZ,),
        "Modulation": (MASK_LINES,),
    }
    emits = {"intensity": ImageOutput}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        raster = Raster.driving(ctx, scan_drives(ctx))
        for _ in count_off(ctx, ctx.inputs[NUM_FRAMES], "frame"):
            ctx.output("intensity", ImageOutput).add(raster.frame(), raster.channel_labels)
