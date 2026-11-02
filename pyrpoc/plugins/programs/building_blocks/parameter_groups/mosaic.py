"""The stage grid a mosaic is imaged on, and the order the stage walks it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.data_library.data import MosaicTile
from pyrpoc.structs.plugins.params import Group, block, float_field, int_field


@block
@dataclass
class MosaicGroup(Group):
    """Tiles along each stage axis and the stage step between them. One
    spacing serves both axes; the display finds the overlap from the images."""

    label: ClassVar[str] = "Mosaic"

    x_tiles: int = int_field("X Tiles", 3, minimum=1, tooltip="Tiles along the stage's x axis")
    y_tiles: int = int_field("Y Tiles", 3, minimum=1, tooltip="Tiles along the stage's y axis")
    spacing_um: float = float_field(
        "Spacing (um)",
        100.0,
        minimum=0.001,
        step=10.0,
        decimals=3,
        tooltip="Stage step between neighbouring tiles, in both x and y",
    )

    def snake(self, centre_um: tuple[float, float]) -> tuple[MosaicTile, ...]:
        """The grid centred on ``centre_um``, rows in turn and each the reverse
        of the last, so the stage never makes a long return move."""
        centre_x, centre_y = centre_um
        tiles: list[MosaicTile] = []
        for row in range(self.y_tiles):
            cols = range(self.x_tiles) if row % 2 == 0 else range(self.x_tiles - 1, -1, -1)
            for col in cols:
                x_um = centre_x + (col - (self.x_tiles - 1) / 2) * self.spacing_um
                y_um = centre_y + (row - (self.y_tiles - 1) / 2) * self.spacing_um
                tiles.append(MosaicTile(row, col, x_um, y_um))
        return tuple(tiles)
