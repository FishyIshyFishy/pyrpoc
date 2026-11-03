"""Mosaic simulation: tiles cut from one synthetic slide, no instruments. The
grid is the real mosaic's, so the display's stitching can be exercised on any
machine."""

from __future__ import annotations

import numpy as np

from pyrpoc.core.data import Image, Mosaic, MosaicTile
from pyrpoc.core.program import Program, RunContext, program_registry

from ..building_blocks.functions.run_loops import grow, walk
from ..building_blocks.functions.synthetic_detector import sim_labels, with_noise
from ..building_blocks.parameter_groups import (
    FrameGroup,
    MosaicGroup,
    PacingGroup,
    SpecimenGroup,
)
from ..building_blocks.runners import Single
from .slide import slide_patch


def tile_frame(
    index: int, tile: MosaicTile, *, frame_shape: FrameGroup, specimen: SpecimenGroup, seed: int
) -> np.ndarray:
    """One ``(C, H, W)`` tile centred on the tile's stage position. Each channel
    is its own slide, as each detector sees its own stain."""
    height, width = frame_shape.y_pixels, frame_shape.x_pixels
    top = round(tile.y_um / specimen.um_per_pixel) - height // 2
    left = round(tile.x_um / specimen.um_per_pixel) - width // 2
    frame = np.stack(
        [
            slide_patch(top, left, height, width, seed=seed + channel)
            for channel in range(frame_shape.channels)
        ]
    )
    return with_noise(frame, specimen.noise_level, seed=seed, index=index)


@program_registry.register("mosaic_simulation")
class MosaicSimulation(Program):
    display_name = "Mosaic"
    group = "Simulations"
    order = 21
    uses = []
    params = [FrameGroup, MosaicGroup, SpecimenGroup, PacingGroup]
    emits = {"mosaic": Mosaic}
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        frame_shape = ctx.params[FrameGroup]
        specimen = ctx.params[SpecimenGroup]
        interval_ms = ctx.params[PacingGroup].frame_interval_ms
        # A virtual stage, centred where the real one would start.
        tiles = ctx.params[MosaicGroup].snake((0.0, 0.0))
        labels = sim_labels(frame_shape.channels)
        # Fixed for the run so every tile is cut from the same slide.
        seed = int(np.random.default_rng().integers(2**31))

        mosaic = Mosaic(tiles, ())
        for index, tile in walk(ctx, tiles):
            frame = tile_frame(index, tile, frame_shape=frame_shape, specimen=specimen, seed=seed)
            mosaic = grow(mosaic, Image(frame, labels))
            ctx.publish("mosaic", mosaic)
            ctx.sleep(interval_ms / 1000.0)
