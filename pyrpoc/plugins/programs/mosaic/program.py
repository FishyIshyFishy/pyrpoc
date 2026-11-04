"""Mosaic: a confocal frame at each stop of a stage grid, walked as a snake."""

from __future__ import annotations

from pyrpoc.core.program import (
    Cancelled,
    Program,
    RunContext,
    program_registry,
)
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.plugins.program_io.inputs import (
    MASK_LINES,
    MASKS,
    SAMPLE_RATE_HZ,
    SCAN_GEOMETRY_INPUTS,
    TILE_SPACING_UM,
    X_TILES,
    Y_TILES,
    MosaicGrid,
)
from pyrpoc.plugins.program_io.outputs import Image, Mosaic, MosaicTile

from ..building_blocks.functions.galvo_raster import Raster
from ..building_blocks.functions.run_loops import grow, walk
from ..building_blocks.runners import Single


def move_stage(ctx: RunContext, stage: PriorStage, x_um: float, y_um: float) -> None:
    """Move and wait for the stage to stop; a stopped run interrupts the wait."""
    stage.move_xy_to(x_um, y_um)
    stage.wait_until_stopped("xy", ctx.sleep)


def image_tiles(ctx: RunContext, stage: PriorStage, tiles: tuple[MosaicTile, ...]) -> None:
    """Image every tile, publishing the mosaic so far after each one."""
    # So the panel fills in live.
    raster = Raster.for_run(ctx)
    mosaic = Mosaic(tiles, ())
    for _, tile in walk(ctx, tiles):
        move_stage(ctx, stage, tile.x_um, tile.y_um)
        mosaic = grow(mosaic, Image(raster.frame(), raster.channel_labels))
        ctx.publish("mosaic", mosaic)


@program_registry.register("mosaic")
class StageMosaic(Program):
    display_name = "Mosaic"
    group = "Confocal Fluorescence"
    order = 12
    devices = [Galvo, DAQ, PriorStage]
    sections = [
        InputSection("Scan", SCAN_GEOMETRY_INPUTS),
        InputSection("DAQ", (SAMPLE_RATE_HZ,)),
        InputSection("Modulation", (MASKS, MASK_LINES)),
        InputSection("Mosaic", (X_TILES, Y_TILES, TILE_SPACING_UM)),
    ]
    emits = {"mosaic": Mosaic}
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
