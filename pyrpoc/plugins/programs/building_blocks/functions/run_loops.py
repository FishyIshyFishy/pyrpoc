"""Loops a run steps through: reporting each step to the status line and
stopping when the run is stopped."""

from __future__ import annotations

from collections.abc import Iterator

from pyrpoc.structs.plugins.programs.program import RunContext

from ..parameter_groups.mosaic import MosaicGroup, Tile


def count_off(ctx: RunContext, count: int, noun: str) -> Iterator[int]:
    """``range(count)``, reporting each step and stopping when the run is stopped."""
    for index in range(count):
        ctx.check_cancel()
        ctx.status(f"{noun} {index + 1}/{count}")
        yield index


def plan_tiles(ctx: RunContext, output: str, centre_um: tuple[float, float]) -> list[Tile]:
    """The run's snake, recorded on ``output`` before the first tile so a
    recording cut short still says where its tiles are."""
    grid = ctx.params[MosaicGroup]
    tiles = grid.snake(centre_um)
    ctx.describe(output, mosaic=grid.layout_metadata(tiles))
    return tiles


def walk(ctx: RunContext, tiles: list[Tile]) -> Iterator[Tile]:
    """The tiles in order, reporting each and stopping when the run is stopped."""
    for tile in tiles:
        ctx.check_cancel()
        ctx.status(f"tile {tile.index + 1}/{len(tiles)} (row {tile.row + 1}, col {tile.col + 1})")
        yield tile
