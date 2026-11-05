"""``MosaicOutput``: images taken across a stage grid, and ``MosaicTile``, one stop of it."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pyrpoc.core.program_output import program_output_registry
from pyrpoc.core.recording import Recording

from .building_blocks.tiff import read_pages
from .image import ImageOutput

# The ``paths`` key of the grid, beside one key per channel.
TILES_KEY = "tiles"


@dataclass(frozen=True)
class MosaicTile:
    """One stop of a mosaic's stage grid: ``row`` grows along stage y and ``col`` along x."""

    row: int
    col: int
    x_um: float
    y_um: float


@program_output_registry.register("mosaic")
class MosaicOutput(ImageOutput):
    """An ``ImageOutput`` taken across a grid: ``tiles`` is the whole plan in order, and image
    ``i`` was taken at ``tiles[i]``. The grid is saved as ``<root>_<name>_tiles.json``, before
    the first image, so a mosaic cut short still says where its images belong."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.tiles: tuple[MosaicTile, ...] = ()

    def plan(self, tiles: tuple[MosaicTile, ...]) -> None:
        """Set the grid, before any image is added."""
        with self.lock:
            if self.arrays:
                raise ValueError(f"{self.name} already holds images; plan it first")
            self.tiles = tiles
            if self.root is not None:
                path = self.root.with_name(f"{self.root.name}_{self.name}_tiles.json")
                path.write_text(json.dumps([asdict(tile) for tile in tiles], indent=2))
                self.paths[TILES_KEY] = path
        self.changed()

    def add(self, array: np.ndarray, channels: Sequence[str]) -> None:
        if len(self) >= len(self.tiles):
            raise ValueError(f"{self.name} has {len(self.tiles)} tiles, all imaged")
        super().add(array, channels)

    @classmethod
    def load(cls, *, name: str, recording: Recording, files: dict[str, Path]) -> MosaicOutput:
        output = cls(name=name, recording=recording, root=None)
        raw = json.loads(files[TILES_KEY].read_text(encoding="utf-8"))
        output.tiles = tuple(
            MosaicTile(int(t["row"]), int(t["col"]), float(t["x_um"]), float(t["y_um"]))
            for t in raw
        )
        channels = {key: path for key, path in files.items() if key != TILES_KEY}
        if channels:
            output.fill(read_pages(channels), list(channels))
        return output
