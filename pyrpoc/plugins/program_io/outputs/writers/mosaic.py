"""A mosaic's tiles as TIFF pages, its overview as TIFFs of its own, and where they all sit
in a JSON beside them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.saving import Writer, writer_registry
from pyrpoc.plugins.program_io.outputs.arrays import Image
from pyrpoc.plugins.program_io.outputs.mosaic import Mosaic, Overview, PlacedTile

from .tiff import append_pages, read_pages

# The ``files`` key of the layout; tile and overview files are keyed by these prefixes.
LAYOUT_KEY = "layout"
TILE_PREFIX = "tile:"
OVERVIEW_PREFIX = "overview:"


def named_paths(root: Path, middle: str, channels: tuple[str, ...]) -> dict[str, Path]:
    """``<root>_<middle>_<channel>.tiff`` for each channel, emptied."""
    paths = {label: root.with_name(f"{root.name}_{middle}_{label}.tiff") for label in channels}
    for path in paths.values():
        path.unlink(missing_ok=True)
    return paths


def layout_of(mosaic: Mosaic) -> dict[str, Any]:
    overview = mosaic.overview
    return {
        "um_per_px": mosaic.um_per_px,
        "planned": mosaic.planned,
        "tiles": [{"left_um": t.left_um, "top_um": t.top_um} for t in mosaic.tiles],
        "overview": None
        if overview is None
        else {
            "left_um": overview.left_um,
            "top_um": overview.top_um,
            "um_per_px": overview.um_per_px,
        },
    }


@writer_registry.register(Mosaic.name)
class MosaicWriter(Writer[Mosaic]):
    """``<root>_<output>_layout.json``, rewritten on each publish;
    ``<root>_<output>_<channel>.tiff``, one page per tile, appended as tiles arrive; and
    ``<root>_<output>_overview_<channel>.tiff`` once the overview is made."""

    def __init__(self, root: Path, output: str, parameters: dict[str, Any]):
        super().__init__(root, output, parameters)
        self.tile_files: dict[str, Path] = {}
        self.tiles_written = 0
        self.overview_written = False

    def write(self, program_output: ProgramOutput[Mosaic], value: Mosaic) -> None:
        layout = self.root.with_name(f"{self.root.name}_{self.output}_layout.json")
        for tile in value.tiles[self.tiles_written :]:
            if not self.tile_files:
                self.tile_files = named_paths(self.root, self.output, value.channels)
                self.paths |= {TILE_PREFIX + key: path for key, path in self.tile_files.items()}
            append_pages(self.tile_files, tile.image)
        self.tiles_written = len(value.tiles)
        if value.overview is not None and not self.overview_written:
            self.write_overview(value.overview)
        # Written last, so it never names a tile that is not on disk yet.
        layout.write_text(json.dumps(layout_of(value), indent=2), encoding="utf-8")
        self.paths[LAYOUT_KEY] = layout

    def write_overview(self, overview: Overview) -> None:
        middle = f"{self.output}_overview"
        files = named_paths(self.root, middle, tuple(overview.image.channels))
        for path, plane in zip(files.values(), overview.image.array, strict=True):
            tifffile.imwrite(str(path), plane)
        self.paths |= {OVERVIEW_PREFIX + key: path for key, path in files.items()}
        self.overview_written = True

    @classmethod
    def read(cls, kind: type[Mosaic], files: dict[str, Path]) -> Mosaic:
        layout = json.loads(files[LAYOUT_KEY].read_text(encoding="utf-8"))
        tile_files = cls.prefixed(files, TILE_PREFIX)
        images = read_pages(tile_files) if tile_files else []
        tiles = tuple(
            PlacedTile(image, float(raw["left_um"]), float(raw["top_um"]))
            for image, raw in zip(images, layout["tiles"], strict=False)
        )
        overview = None
        if layout["overview"] is not None:
            overview_files = cls.prefixed(files, OVERVIEW_PREFIX)
            planes = [tifffile.imread(str(path)) for path in overview_files.values()]
            raw = layout["overview"]
            overview = Overview(
                Image(np.stack(planes), list(overview_files)),
                float(raw["left_um"]),
                float(raw["top_um"]),
                float(raw["um_per_px"]),
            )
        return Mosaic(float(layout["um_per_px"]), int(layout["planned"]), tiles, overview)

    @staticmethod
    def prefixed(files: dict[str, Path], prefix: str) -> dict[str, Path]:
        """The files whose key starts with ``prefix``, keyed by the rest: the channel."""
        return {key[len(prefix) :]: path for key, path in files.items() if key.startswith(prefix)}
