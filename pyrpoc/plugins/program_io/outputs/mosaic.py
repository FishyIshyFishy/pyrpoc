"""``Mosaic``: images taken across a stage grid, and ``MosaicTile``, one stop of it."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.program_output import Data, data_kind_registry

from .arrays import Image


@dataclass(frozen=True)
class MosaicTile:
    """One stop of a mosaic's stage grid: ``row`` grows along stage y and
    ``col`` along x."""

    row: int
    col: int
    x_um: float
    y_um: float


@data_kind_registry.register("Mosaic")
@dataclass(frozen=True, eq=False)
class Mosaic(Data):
    """Images taken across a stage grid: ``tiles`` is the whole plan in order,
    and ``images[i]`` was taken at ``tiles[i]``."""

    tiles: tuple[MosaicTile, ...]
    images: tuple[Image, ...]

    def __post_init__(self) -> None:
        if len(self.images) > len(self.tiles):
            raise ValueError(f"a mosaic of {len(self.tiles)} tiles got {len(self.images)} images")

    @property
    def nbytes(self) -> int:
        return sum(image.nbytes for image in self.images)

    def summary(self) -> str:
        return f"{len(self.images)}/{len(self.tiles)} tiles"
