"""The mosaic grid: tiles along each stage axis and the step between them, and
``MosaicGrid``, which lays out the tiles."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.plugins.program_io.outputs.mosaic import MosaicTile
from pyrpoc.utils.value_rules import float_input, int_input

X_TILES = int_input("x_tiles", "X Tiles", 3, minimum=1, tooltip="Tiles along the stage's x axis")
Y_TILES = int_input("y_tiles", "Y Tiles", 3, minimum=1, tooltip="Tiles along the stage's y axis")
# One spacing serves both axes; the display finds the overlap from the images.
TILE_SPACING_UM = float_input(
    "tile_spacing_um",
    "Spacing (um)",
    100.0,
    minimum=0.001,
    step=10.0,
    decimals=3,
    tooltip="Stage step between neighbouring tiles, in both x and y",
)


@dataclass(frozen=True)
class MosaicGrid:
    """Tiles along each stage axis and the stage step between them."""

    x_tiles: int
    y_tiles: int
    spacing_um: float

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> MosaicGrid:
        return cls(values[X_TILES], values[Y_TILES], values[TILE_SPACING_UM])

    def snake(self, centre_um: tuple[float, float]) -> tuple[MosaicTile, ...]:
        """The grid centred on ``centre_um``, rows in turn and each the reverse
        of the last."""
        # Reversing each row means the stage never makes a long return move.
        centre_x, centre_y = centre_um
        tiles: list[MosaicTile] = []
        for row in range(self.y_tiles):
            cols = range(self.x_tiles) if row % 2 == 0 else range(self.x_tiles - 1, -1, -1)
            for col in cols:
                x_um = centre_x + (col - (self.x_tiles - 1) / 2) * self.spacing_um
                y_um = centre_y + (row - (self.y_tiles - 1) / 2) * self.spacing_um
                tiles.append(MosaicTile(row, col, x_um, y_um))
        return tuple(tiles)
