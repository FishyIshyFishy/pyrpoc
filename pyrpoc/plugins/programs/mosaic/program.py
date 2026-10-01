"""Mosaic: a confocal frame at each stop of a stage grid, walked as a snake."""

from __future__ import annotations

from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.structs.data_library.data import Image2D
from pyrpoc.structs.plugins.programs.program import (
    Cancelled,
    Program,
    RunContext,
    program_registry,
)

from ..building_blocks.functions.galvo_raster import Raster
from ..building_blocks.functions.run_loops import plan_tiles, walk
from ..building_blocks.parameter_groups import (
    DaqGroup,
    ModulationGroup,
    MosaicGroup,
    ScanGroup,
    Tile,
)
from ..building_blocks.runners import Single


def move_stage(ctx: RunContext, stage: PriorStage, x_um: float, y_um: float) -> None:
    """Move and wait for the stage to stop; a stopped run interrupts the wait."""
    stage.move_xy_to(x_um, y_um)
    stage.wait_until_stopped("xy", ctx.sleep)


def image_tiles(ctx: RunContext, stage: PriorStage, tiles: list[Tile]) -> None:
    raster = Raster.for_run(ctx)
    for tile in walk(ctx, tiles):
        move_stage(ctx, stage, tile.x_um, tile.y_um)
        ctx.publish("intensity", raster.frame(), channels=raster.channel_labels)


@program_registry.register("mosaic")
class Mosaic(Program):
    display_name = "Mosaic"
    group = "Confocal Fluorescence"
    order = 12
    uses = [Galvo, DAQ, PriorStage]
    params = [ScanGroup, DaqGroup, ModulationGroup, MosaicGroup]
    emits = {"intensity": Image2D}
    # Once only: a mosaic is one pass over the grid, never repeated or looped.
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        stage = ctx.devices[PriorStage]
        origin = stage.xy_position()
        tiles = plan_tiles(ctx, "intensity", origin)
        try:
            image_tiles(ctx, stage, tiles)
        except Cancelled:
            stage.stop_smoothly()
            raise
        # Back where it started, so a repeat run images the same area.
        ctx.status("returning to the start position")
        move_stage(ctx, stage, *origin)
