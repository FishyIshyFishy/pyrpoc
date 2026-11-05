"""``Mosaic``: tiles placed on the dish by position, and the overview stitched from them."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pyrpoc.core.program_output import Data, data_kind_registry

from .arrays import Image


@dataclass(frozen=True)
class PlacedTile:
    """One tile's image, oriented as the dish is, and the dish position of its top-left
    pixel, in µm."""

    image: Image
    left_um: float
    top_um: float


@dataclass(frozen=True)
class Overview:
    """One image of the whole mosaic and the dish position of its top-left pixel, in µm, at
    ``um_per_px``."""

    image: Image
    left_um: float
    top_um: float
    um_per_px: float


@data_kind_registry.register("Mosaic")
@dataclass(frozen=True, eq=False)
class Mosaic(Data):
    """Tiles of one pixel size placed on the dish as they arrive, ``planned`` of them in all,
    and once stitched, the overview."""

    um_per_px: float
    planned: int
    tiles: tuple[PlacedTile, ...]
    overview: Overview | None

    @property
    def nbytes(self) -> int:
        tiles = sum(tile.image.nbytes for tile in self.tiles)
        return tiles + (0 if self.overview is None else self.overview.image.nbytes)

    def summary(self) -> str:
        stitched = ", stitched" if self.overview is not None else ""
        return f"{len(self.tiles)}/{self.planned} tiles{stitched}"

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self.tiles[0].image.channels) if self.tiles else ()

    def extent_um(self) -> tuple[float, float, float, float]:
        """(left, top, right, bottom) of every tile, in dish µm."""
        _, height, width = self.tiles[0].image.array.shape
        return (
            min(tile.left_um for tile in self.tiles),
            min(tile.top_um for tile in self.tiles),
            max(tile.left_um for tile in self.tiles) + width * self.um_per_px,
            max(tile.top_um for tile in self.tiles) + height * self.um_per_px,
        )

    def with_tile(self, tile: PlacedTile) -> Mosaic:
        return replace(self, tiles=(*self.tiles, tile))

    def with_overview(self, overview: Overview) -> Mosaic:
        return replace(self, overview=overview)
