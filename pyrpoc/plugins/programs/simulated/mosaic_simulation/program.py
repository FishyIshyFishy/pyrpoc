"""Mosaic simulation: tiles cut from one synthetic slide, with no instruments."""

# The grid is the real mosaic's, so the display's stitching can be exercised on any machine.

from __future__ import annotations

import numpy as np

from pyrpoc.core.program import Program, RunContext
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.program_io.inputs import (
    TILE_SPACING_UM,
    X_PIXELS,
    X_TILES,
    Y_PIXELS,
    Y_TILES,
    MosaicGrid,
)
from pyrpoc.plugins.program_io.outputs import MosaicOutput, MosaicTile

from ...building_blocks.functions.run_loops import walk
from ...building_blocks.runners import Single
from ..frame import SIMULATED_CHANNELS, FrameShape
from ..pacing import FRAME_INTERVAL_MS
from ..specimen import UM_PER_PIXEL, SyntheticSlide
from ..synthetic_detector import NOISE_LEVEL, sim_labels, with_noise
from .slide import slide_patch


def tile_frame(
    index: int, tile: MosaicTile, *, frame_shape: FrameShape, specimen: SyntheticSlide, seed: int
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


class MosaicSimulation(Program):
    display_name = "Mosaic"
    group = "Simulations"
    order = 21
    devices = []
    sections = [
        InputSection("Frame", (X_PIXELS, Y_PIXELS, SIMULATED_CHANNELS)),
        InputSection("Mosaic", (X_TILES, Y_TILES, TILE_SPACING_UM)),
        InputSection("Specimen", (UM_PER_PIXEL, NOISE_LEVEL)),
        InputSection("Pacing", (FRAME_INTERVAL_MS,)),
    ]
    emits = {"mosaic": MosaicOutput}
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        frame_shape = FrameShape.from_inputs(ctx.inputs)
        specimen = SyntheticSlide.from_inputs(ctx.inputs)
        interval_ms = ctx.inputs[FRAME_INTERVAL_MS]
        # A virtual stage, centred where the real one would start.
        tiles = MosaicGrid.from_inputs(ctx.inputs).snake((0.0, 0.0))
        labels = sim_labels(frame_shape.channels)
        # Fixed for the run so every tile is cut from the same slide.
        seed = int(np.random.default_rng().integers(2**31))

        mosaic = ctx.output("mosaic", MosaicOutput)
        mosaic.plan(tiles)
        for index, tile in walk(ctx, tiles):
            frame = tile_frame(index, tile, frame_shape=frame_shape, specimen=specimen, seed=seed)
            mosaic.add(frame, labels)
            ctx.sleep(interval_ms / 1000.0)
