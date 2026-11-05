"""Where each of a mosaic's images sits on its grid, as the views place them.
Only each tile's image index and grid cell; stage positions are kept on disk
for people, not used here."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.plugins.program_io.outputs import MosaicOutput


@dataclass(frozen=True)
class GridTile:
    """Image ``index`` of the mosaic sits in grid cell (``row``, ``col``)."""

    index: int
    row: int
    col: int


@dataclass(frozen=True)
class MosaicLayout:
    tiles: tuple[GridTile, ...]


def layout_of(mosaic: MosaicOutput) -> MosaicLayout:
    return MosaicLayout(
        tuple(GridTile(index, tile.row, tile.col) for index, tile in enumerate(mosaic.tiles))
    )
