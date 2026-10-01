"""Stitching recovers the stage step from tile content alone."""

from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.plugins.data_panels.mosaic.layout import MosaicLayout, read_layout
from pyrpoc.plugins.data_panels.mosaic.stitching import Stitcher, composite, tile_positions
from pyrpoc.plugins.programs.building_blocks.parameter_groups import MosaicGroup

TILE = 96
FALLBACK = 0.1


def snake_layout(x_tiles: int, y_tiles: int) -> MosaicLayout:
    """The layout the panel reads back from a snake the program walked."""
    grid = MosaicGroup(x_tiles=x_tiles, y_tiles=y_tiles)
    layout = read_layout({"mosaic": grid.layout_metadata(grid.snake((0.0, 0.0)))})
    assert layout is not None
    return layout


def texture(size: int) -> np.ndarray:
    """Smooth random structure: white noise low-passed in frequency."""
    noise = np.random.default_rng(3).standard_normal((size, size))
    freq = np.fft.fftfreq(size)
    lowpass = np.exp(-(freq[:, None] ** 2 + freq[None, :] ** 2) / (2 * 0.04**2))
    return np.fft.ifft2(np.fft.fft2(noise) * lowpass).real.astype(np.float32)


def cut_tiles(
    source: np.ndarray, layout: MosaicLayout, col_step: tuple[int, int], row_step: tuple[int, int]
) -> list[np.ndarray]:
    """``(1, H, W)`` frames in acquisition order, from a corner well inside ``source``."""
    frames = []
    for tile in layout.tiles:
        top = 200 + tile.col * col_step[0] + tile.row * row_step[0]
        left = 200 + tile.col * col_step[1] + tile.row * row_step[1]
        frames.append(source[top : top + TILE, left : left + TILE][None])
    return frames


@pytest.mark.parametrize(
    ("col_step", "row_step"),
    [
        ((0, 80), (80, 0)),
        ((0, 70), (70, 0)),
        # The stage's x runs against the scan's.
        ((0, -80), (80, 0)),
        # The stage is slightly rotated from the scan axes.
        ((3, 80), (80, -2)),
    ],
)
def test_steps_are_recovered(col_step: tuple[int, int], row_step: tuple[int, int]) -> None:
    layout = snake_layout(3, 3)
    frames = cut_tiles(texture(640), layout, col_step, row_step)

    steps = Stitcher(layout).steps(frames, FALLBACK)

    assert (steps.col.offset, steps.row.offset) == (col_step, row_step)
    assert (steps.col.confident, steps.row.confident) == (6, 6)


def test_the_composite_reproduces_the_specimen() -> None:
    source = texture(640)
    layout = snake_layout(3, 2)
    frames = cut_tiles(source, layout, (0, 80), (80, 0))

    steps = Stitcher(layout).steps(frames, FALLBACK)
    positions = tile_positions(layout, steps, len(frames))
    image = composite(frames, positions)

    assert image.shape == (1, 80 + TILE, 2 * 80 + TILE)
    np.testing.assert_allclose(image[0], source[200 : 200 + 176, 200 : 200 + 256], atol=1e-4)


def test_tiles_so_far_are_placed_during_a_run() -> None:
    layout = snake_layout(3, 3)
    frames = cut_tiles(texture(640), layout, (0, 80), (80, 0))[:4]

    steps = Stitcher(layout).steps(frames, FALLBACK)
    positions = tile_positions(layout, steps, len(frames))

    # Tiles 0-2 are row 0; tile 3 is row 1, col 2, reached by the snake.
    assert (steps.col.registered, steps.row.registered) == (2, 1)
    assert positions == {0: (0, 0), 1: (0, 80), 2: (0, 160), 3: (80, 160)}


def test_featureless_tiles_fall_back_to_the_nominal_overlap() -> None:
    layout = snake_layout(2, 2)
    frames = [np.full((1, TILE, TILE), 0.5, dtype=np.float32) for _ in layout.tiles]

    steps = Stitcher(layout).steps(frames, FALLBACK)

    assert steps.col.nominal and steps.row.nominal
    assert (steps.col.offset, steps.row.offset) == ((0, 86), (86, 0))


def test_a_pair_registers_on_the_channel_with_structure() -> None:
    layout = snake_layout(3, 2)
    textured = cut_tiles(texture(640), layout, (0, 80), (80, 0))
    blank = np.zeros((1, TILE, TILE), dtype=np.float32)
    frames = [np.concatenate([blank, tile]) for tile in textured]

    steps = Stitcher(layout).steps(frames, FALLBACK)

    assert (steps.col.offset, steps.row.offset) == ((0, 80), (80, 0))
