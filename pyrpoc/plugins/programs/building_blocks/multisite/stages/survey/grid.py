"""The tiles that cover a region, walked as a snake."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    """One stop: its place in the walk and on the grid, and the stage position, in µm, that
    centres the field on it."""

    index: int
    row: int
    col: int
    x_um: float
    y_um: float


def count_along(extent_um: float, field_um: float, step_um: float) -> int:
    """How many fields, ``step_um`` apart, cover ``extent_um``."""
    return max(1, math.ceil((extent_um - field_um) / step_um - 1e-9) + 1)


def snake(
    corner: tuple[float, float], size: tuple[float, float], field_um: float, overlap: float
) -> tuple[Tile, ...]:
    """Tiles covering ``size`` (width, height) from its top-left ``corner``, row by row, each
    row walked the other way from the last so the stage never runs back."""
    step = field_um * (1.0 - overlap)
    columns = count_along(size[0], field_um, step)
    rows = count_along(size[1], field_um, step)
    tiles = []
    for row in range(rows):
        order = range(columns) if row % 2 == 0 else range(columns - 1, -1, -1)
        for col in order:
            x = corner[0] + field_um / 2.0 + col * step
            y = corner[1] + field_um / 2.0 + row * step
            tiles.append(Tile(len(tiles), row, col, x, y))
    return tuple(tiles)
