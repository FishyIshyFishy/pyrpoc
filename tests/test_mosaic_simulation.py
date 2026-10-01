"""A simulated mosaic, recorded and reloaded as the real program's would be."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyrpoc.data_library.format import load_recording
from pyrpoc.plugins.data_panels.mosaic.layout import GridTile, read_layout
from pyrpoc.plugins.programs.building_blocks.parameter_groups import (
    FrameGroup,
    MosaicGroup,
    PacingGroup,
    SpecimenGroup,
)
from pyrpoc.plugins.programs.mosaic_simulation.program import MosaicSimulation
from pyrpoc.structs.plugins.params import BlockStore

from .helpers import Recorder


def small_mosaic_blocks() -> BlockStore:
    """3 x 2 tiles of 64 px, 48 px apart: 16 px of overlap between neighbours."""
    blocks = BlockStore()
    frame = blocks.get(FrameGroup)
    frame.x_pixels = frame.y_pixels = 64
    frame.channels = 1
    grid = blocks.get(MosaicGroup)
    grid.x_tiles, grid.y_tiles, grid.spacing_um = 3, 2, 24.0
    specimen = blocks.get(SpecimenGroup)
    specimen.um_per_pixel, specimen.noise_level = 0.5, 0.0
    blocks.get(PacingGroup).frame_interval_ms = 0
    return blocks


def test_a_recorded_mosaic_reloads_with_its_layout(tmp_path: Path) -> None:
    (dataset,) = Recorder().record(
        MosaicSimulation(), "mosaic_simulation", small_mosaic_blocks(), tmp_path, "tiles"
    )
    assert dataset.meta_path is not None
    (loaded,) = load_recording(dataset.meta_path)

    layout = read_layout(loaded.metadata)
    assert layout is not None
    walked = MosaicGroup(x_tiles=3, y_tiles=2).snake((0.0, 0.0))
    assert layout.tiles == tuple(GridTile(tile.index, tile.row, tile.col) for tile in walked)
    assert len(loaded) == 6


def test_neighbouring_tiles_share_their_overlap(tmp_path: Path) -> None:
    (dataset,) = Recorder().record(
        MosaicSimulation(), "mosaic_simulation", small_mosaic_blocks(), tmp_path, "tiles"
    )
    frames = dataset.frames()

    # Tile 1 is one column right of tile 0; tile 4 is one row below tile 1.
    np.testing.assert_array_equal(frames[0][:, :, 48:], frames[1][:, :, :16])
    np.testing.assert_array_equal(frames[1][:, 48:, :], frames[4][:, :16, :])
