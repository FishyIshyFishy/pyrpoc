"""The mosaic layout: the snake the program walks, and the panel reading it
back from metadata. The two share only the documented metadata shape."""

from __future__ import annotations

import pytest

from pyrpoc.plugins.data_panels.mosaic.layout import GridTile, read_layout
from pyrpoc.plugins.programs.building_blocks.parameter_groups import MosaicGroup


def test_snake_reverses_every_other_row() -> None:
    tiles = MosaicGroup(x_tiles=3, y_tiles=2).snake((0.0, 0.0))

    assert [(tile.row, tile.col) for tile in tiles] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 1),
        (1, 0),
    ]
    assert [tile.index for tile in tiles] == list(range(6))


def test_a_single_row_runs_left_to_right() -> None:
    tiles = MosaicGroup(x_tiles=4, y_tiles=1).snake((0.0, 0.0))

    assert [tile.col for tile in tiles] == [0, 1, 2, 3]


def test_the_grid_is_centred_on_the_start() -> None:
    tiles = MosaicGroup(x_tiles=3, y_tiles=3, spacing_um=10.0).snake((100.0, 50.0))

    first, middle = tiles[0], tiles[4]
    assert (first.x_um, first.y_um) == (90.0, 40.0)
    assert (middle.row, middle.col, middle.x_um, middle.y_um) == (1, 1, 100.0, 50.0)


def test_the_panel_reads_what_the_program_writes() -> None:
    grid = MosaicGroup(x_tiles=2, y_tiles=3, spacing_um=12.5)
    tiles = grid.snake((1.0, -2.0))

    layout = read_layout({"mosaic": grid.layout_metadata(tiles)})

    assert layout is not None
    assert layout.tiles == tuple(GridTile(tile.index, tile.row, tile.col) for tile in tiles)


def test_metadata_without_a_layout_is_not_a_mosaic() -> None:
    assert read_layout({"units": "counts"}) is None


def test_a_malformed_layout_raises() -> None:
    with pytest.raises(ValueError, match="malformed"):
        read_layout({"mosaic": {"path": "snake", "tiles": [{}]}})
