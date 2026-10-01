"""Mosaic simulation: tiles cut from one endless synthetic slide, no instruments.

The slide is a function of absolute pixel position, so neighbouring tiles
share the features in their overlap exactly as a real specimen would, and the
display's stitching can be exercised on any machine. Blobs are seeded per
lattice cell rather than drawn on one canvas, so a wide grid costs no more
memory than one tile. The grid and its metadata are the real program's.
"""

from __future__ import annotations

import numpy as np

from pyrpoc.structs.data_library.data import Image2D
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from .components.param_groups import FrameGroup, MosaicGroup, PacingGroup, SpecimenGroup, Tile
from .components.runners import Single

# The slide's lattice: each cell holds its own blobs, drawn from its own seed.
CELL_PIXELS = 48
BLOBS_PER_CELL = 3
SIGMA_PIXELS = (1.5, 6.0)
# A blob is negligible beyond this many sigmas, so cells this far out are skipped.
REACH_SIGMAS = 3.0
NOISE_STREAM = 0x0125E


def _rng(*parts: int) -> np.random.Generator:
    return np.random.default_rng([part % (2**32) for part in parts])


def add_cell_blobs(
    plane: np.ndarray, top: int, left: int, *, cell_y: int, cell_x: int, seed: int
) -> None:
    """Add one lattice cell's blobs to ``plane``, whose pixel (0, 0) is slide
    pixel (``top``, ``left``). Each blob only touches the window it reaches."""
    generator = _rng(seed, cell_y, cell_x)
    centre_y = cell_y * CELL_PIXELS + generator.uniform(0.0, CELL_PIXELS, BLOBS_PER_CELL)
    centre_x = cell_x * CELL_PIXELS + generator.uniform(0.0, CELL_PIXELS, BLOBS_PER_CELL)
    sigma = generator.uniform(*SIGMA_PIXELS, BLOBS_PER_CELL)
    amplitude = generator.uniform(0.3, 1.0, BLOBS_PER_CELL)

    height, width = plane.shape
    for cy, cx, s, a in zip(centre_y, centre_x, sigma, amplitude, strict=True):
        reach = REACH_SIGMAS * s
        y0, y1 = max(0, int(cy - reach) - top), min(height, int(cy + reach) + 1 - top)
        x0, x1 = max(0, int(cx - reach) - left), min(width, int(cx + reach) + 1 - left)
        if y0 >= y1 or x0 >= x1:
            continue
        ys = np.arange(y0 + top, y1 + top, dtype=np.float32) - cy
        xs = np.arange(x0 + left, x1 + left, dtype=np.float32) - cx
        spread = 2.0 * s**2
        plane[y0:y1, x0:x1] += a * np.outer(np.exp(-(ys**2) / spread), np.exp(-(xs**2) / spread))


def slide_patch(top: int, left: int, height: int, width: int, *, seed: int) -> np.ndarray:
    """The slide's ``(height, width)`` window starting at pixel (``top``, ``left``)."""
    margin = int(REACH_SIGMAS * SIGMA_PIXELS[1]) + 1
    plane = np.zeros((height, width), dtype=np.float32)
    for cell_y in range((top - margin) // CELL_PIXELS, (top + height + margin) // CELL_PIXELS + 1):
        for cell_x in range(
            (left - margin) // CELL_PIXELS, (left + width + margin) // CELL_PIXELS + 1
        ):
            add_cell_blobs(plane, top, left, cell_y=cell_y, cell_x=cell_x, seed=seed)
    return np.clip(plane, 0.0, 1.0)


def tile_frame(
    tile: Tile, *, frame_shape: FrameGroup, specimen: SpecimenGroup, seed: int
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
    if specimen.noise_level:
        noise = _rng(seed, tile.index, NOISE_STREAM).standard_normal(frame.shape)
        frame = frame + noise.astype(np.float32) * specimen.noise_level
    return np.clip(frame, 0.0, None).astype(np.float32, copy=False)


@program_registry.register("mosaic_simulation")
class MosaicSimulation(Program):
    display_name = "Mosaic Simulation"
    uses = []
    params = [FrameGroup, MosaicGroup, SpecimenGroup, PacingGroup]
    emits = {"intensity": Image2D}
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        frame_shape = ctx.params[FrameGroup]
        grid = ctx.params[MosaicGroup]
        specimen = ctx.params[SpecimenGroup]
        interval_ms = ctx.params[PacingGroup].frame_interval_ms

        # A virtual stage, centred where the real one would start.
        tiles = grid.snake((0.0, 0.0))
        ctx.describe("intensity", mosaic=grid.layout_metadata(tiles))
        labels = [f"sim{index}" for index in range(frame_shape.channels)]
        # Fixed for the run so every tile is cut from the same slide.
        seed = int(np.random.default_rng().integers(2**31))

        for tile in tiles:
            ctx.check_cancel()
            ctx.status(
                f"tile {tile.index + 1}/{len(tiles)} (row {tile.row + 1}, col {tile.col + 1})"
            )
            frame = tile_frame(tile, frame_shape=frame_shape, specimen=specimen, seed=seed)
            ctx.publish("intensity", frame, channels=labels)
            ctx.sleep(interval_ms / 1000.0)
