"""A mosaic's images as TIFF pages, and its grid beside them."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pyrpoc.core.data import Mosaic, MosaicTile
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.saving import Writer, writer_registry

from .tiff import append_pages, channel_paths, read_pages

# The ``files`` key of the grid, beside one key per channel.
TILES_KEY = "tiles"


@writer_registry.register(Mosaic.name)
class MosaicWriter(Writer[Mosaic]):
    """``<root>_<output>_tiles.json``, the whole grid, written before the first
    image so a mosaic cut short still says where its images belong; then
    ``<root>_<channel>.tiff`` with page ``i`` taken at tile ``i``. Each publish
    is the mosaic so far, so only images not yet on disk are appended."""

    def __init__(self, root: Path, output: str, parameters: dict[str, Any]):
        super().__init__(root, output, parameters)
        self.channel_files: dict[str, Path] = {}
        self.images_written = 0

    def write(self, program_output: ProgramOutput[Mosaic], value: Mosaic) -> None:
        if not self.paths:
            tiles = self.root.with_name(f"{self.root.name}_{self.output}_tiles.json")
            tiles.write_text(json.dumps([asdict(tile) for tile in value.tiles], indent=2))
            self.paths[TILES_KEY] = tiles
        for image in value.images[self.images_written :]:
            if not self.channel_files:
                self.channel_files = channel_paths(self.root, image)
                self.paths |= self.channel_files
            append_pages(self.channel_files, image)
        self.images_written = len(value.images)

    @classmethod
    def read(cls, kind: type[Mosaic], files: dict[str, Path]) -> Mosaic:
        channels = {key: path for key, path in files.items() if key != TILES_KEY}
        raw = json.loads(files[TILES_KEY].read_text(encoding="utf-8"))
        tiles = tuple(
            MosaicTile(int(t["row"]), int(t["col"]), float(t["x_um"]), float(t["y_um"]))
            for t in raw
        )
        images = tuple(read_pages(channels)) if channels else ()
        return Mosaic(tiles, images)
