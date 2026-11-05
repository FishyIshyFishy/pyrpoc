"""Mosaic: a confocal frame at each stop of a stage grid, walked as a snake."""

from __future__ import annotations

from pyrpoc.core.program import (
    Cancelled,
    Program,
    RunContext,
    program_registry,
)
from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.plugins.program_io.inputs import (
    MASK_LINES,
    MASKS,
    SAMPLE_RATE_MHZ,
    SCAN_GEOMETRY_INPUTS,
    TILE_SPACING_UM,
    X_TILES,
    Y_TILES,
    MosaicGrid,
)
from pyrpoc.plugins.program_io.outputs import MosaicOutput, MosaicTile

from ..building_blocks.functions.galvo_raster import Raster
from ..building_blocks.functions.run_loops import walk
from ..building_blocks.runners import Single


def move_stage(ctx: RunContext, stage: PriorStage, x_um: float, y_um: float) -> None:
    """Move and wait for the stage to stop; a stopped run interrupts the wait."""
    stage.move_xy_to(x_um, y_um)
    stage.wait_until_stopped("xy", ctx.sleep)


def image_tiles(ctx: RunContext, stage: PriorStage, tiles: tuple[MosaicTile, ...]) -> None:
    """Image every tile, adding each image to the mosaic as it is taken."""
    raster = Raster.for_run(ctx)
    mosaic = ctx.output("mosaic", MosaicOutput)
    mosaic.plan(tiles)
    for _, tile in walk(ctx, tiles):
        move_stage(ctx, stage, tile.x_um, tile.y_um)
        mosaic.add(raster.frame(), raster.channel_labels)


@program_registry.register("mosaic")
class StageMosaic(Program):
    display_name = "Mosaic"
    group = "Conventional"
    order = 120
    devices = [Galvo, DAQ, PriorStage]
    inputs = [
        *SCAN_GEOMETRY_INPUTS,
        SAMPLE_RATE_MHZ,
        MASKS,
        MASK_LINES,
        X_TILES,
        Y_TILES,
        TILE_SPACING_UM,
    ]
    # MASKS has no row: masks are drawn and edited in a Mask Editor.
    cards = {
        "Scan": SCAN_GEOMETRY_INPUTS,
        "DAQ": (SAMPLE_RATE_MHZ,),
        "Modulation": (MASK_LINES,),
        "Mosaic": (
            X_TILES,
            Y_TILES,
            TILE_SPACING_UM,
        ),
    }
    emits = {"mosaic": MosaicOutput}
    # Once only: a mosaic is one pass over the grid, never repeated or looped.
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        stage = ctx.devices[PriorStage]
        origin = stage.xy_position()
        tiles = MosaicGrid.from_inputs(ctx.inputs).snake(origin)
        try:
            image_tiles(ctx, stage, tiles)
        except Cancelled:
            stage.stop_smoothly()
            raise
        # Back where it started, so a repeat run images the same area.
        ctx.status("returning to the start position")
        move_stage(ctx, stage, *origin)
