"""Mosaic: a grid of tiles around the stage, scanned with the galvo calibration and blended into
one overview, each where the stage put it."""

from __future__ import annotations

from pyrpoc.core.program import RunContext, program_registry
from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.plugins.program_io.inputs import EXTRA_LEFT, EXTRA_RIGHT, SAMPLE_RATE_MHZ

from ..building_blocks.multisite import Microscope, MosaicProgram, RigMicroscope


@program_registry.register("mosaic")
class StageMosaic(MosaicProgram):
    display_name = "Mosaic"
    group = "Confocal Fluorescence"
    order = 12
    devices = [Galvo, DAQ, PriorStage]
    inputs = [*MosaicProgram.inputs, EXTRA_LEFT, EXTRA_RIGHT, SAMPLE_RATE_MHZ]
    fixed = [*MosaicProgram.fixed, EXTRA_LEFT, EXTRA_RIGHT, SAMPLE_RATE_MHZ]
    cards = {
        **MosaicProgram.cards,
        "Scan": (EXTRA_LEFT, EXTRA_RIGHT),
        "DAQ": (SAMPLE_RATE_MHZ,),
    }

    def microscope(self, ctx: RunContext) -> Microscope:
        return RigMicroscope(ctx)

    def overscan(self, ctx: RunContext) -> tuple[int, int]:
        return (ctx.inputs[EXTRA_LEFT], ctx.inputs[EXTRA_RIGHT])
