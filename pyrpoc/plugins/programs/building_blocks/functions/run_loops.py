"""Loops a run steps through: reporting each step to the status line and
stopping when the run is stopped."""

from __future__ import annotations

from collections.abc import Iterator

from pyrpoc.core.data import Image, Mosaic, MosaicTile
from pyrpoc.core.program import RunContext


def count_off(ctx: RunContext, count: int, noun: str) -> Iterator[int]:
    """``range(count)``, reporting each step and stopping when the run is stopped."""
    for index in range(count):
        ctx.check_cancel()
        ctx.status(f"{noun} {index + 1}/{count}")
        yield index


def walk(ctx: RunContext, tiles: tuple[MosaicTile, ...]) -> Iterator[tuple[int, MosaicTile]]:
    """The tiles in order with their index, reporting each and stopping when
    the run is stopped."""
    for index, tile in enumerate(tiles):
        ctx.check_cancel()
        ctx.status(f"tile {index + 1}/{len(tiles)} (row {tile.row + 1}, col {tile.col + 1})")
        yield index, tile


def grow(mosaic: Mosaic, image: Image) -> Mosaic:
    """``mosaic`` with ``image`` taken at its next tile."""
    return Mosaic(mosaic.tiles, (*mosaic.images, image))
