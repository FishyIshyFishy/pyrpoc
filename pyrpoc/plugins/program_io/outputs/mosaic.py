"""``MosaicOutput``: dish-oriented images taken across a stage grid and stitched into one
overview, with ``MosaicTile``, one stop of the grid, and ``Placement``, where an image lies on
the dish."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

from pyrpoc.core.program_output import program_output_registry
from pyrpoc.core.recording import Recording
from pyrpoc.utils.file_formats.tiff import append_pages, channel_paths, read_pages

from .image import ImageOutput

# The ``paths`` keys of the grid and of each overview channel, beside one key per channel.
GRID_KEY = "grid"
OVERVIEW_PREFIX = "overview:"

M = TypeVar("M", bound="MosaicOutput")


@dataclass(frozen=True)
class MosaicTile:
    """One stop of a mosaic's stage grid: ``row`` grows along stage y and ``col`` along x, and
    (``x_um``, ``y_um``) is the stage position that centres the field on it."""

    row: int
    col: int
    x_um: float
    y_um: float


@dataclass(frozen=True)
class Placement:
    """Where an image lies on the dish: its top-left pixel's position and its pixel size,
    in µm."""

    left_um: float
    top_um: float
    um_per_px: float


@program_output_registry.register("mosaic")
class MosaicOutput(ImageOutput):
    """An ``ImageOutput`` of square, dish-oriented tiles ``um_per_px`` µm a pixel: ``tiles`` is
    the whole grid in order, and image ``i`` was taken at ``tiles[i]``. Once finished it holds
    the overview the tiles were stitched into. The grid is saved as
    ``<root>_<name>_grid.json`` before the first image, so a mosaic cut short still says where
    its images belong; the overview as ``<root>_<name>_overview_<channel>.tiff``."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.tiles: tuple[MosaicTile, ...] = ()
        self.um_per_px = 0.0
        self.overview_planes: np.ndarray | None = None
        self.placement: Placement | None = None

    def plan(self, tiles: tuple[MosaicTile, ...], um_per_px: float) -> None:
        """Set the grid and the pixel size, before any image is added."""
        with self.lock:
            if self.arrays:
                raise ValueError(f"{self.name} already holds images; plan it first")
            self.tiles, self.um_per_px = tiles, um_per_px
            self.write_grid()
        self.changed()

    def add(self, array: np.ndarray, channels: Sequence[str]) -> None:
        if len(self) >= len(self.tiles):
            raise ValueError(f"{self.name} has {len(self.tiles)} tiles, all imaged")
        super().add(array, channels)

    def stitched(self, overview: np.ndarray, placement: Placement) -> None:
        """Take the ``(C, H, W)`` overview the tiles were stitched into, lying at
        ``placement``."""
        planes = np.ascontiguousarray(overview, dtype=np.float32)
        with self.lock:
            self.overview_planes, self.placement = planes, placement
            if self.root is not None:
                self.write_overview(self.root, planes)
            self.write_grid()
        self.changed()

    @property
    def is_stitched(self) -> bool:
        return self.placement is not None

    def overview(self) -> tuple[np.ndarray, Placement]:
        """The overview and where it lies, which a finished mosaic always holds."""
        with self.lock:
            if self.overview_planes is None or self.placement is None:
                raise ValueError(f"{self.name} was not finished, so it has no overview")
            return self.overview_planes, self.placement

    def tile_corner(self, index: int) -> tuple[float, float]:
        """The dish position (left, top) of tile ``index``'s top-left pixel, where the stage
        put it."""
        tile = self.tiles[index]
        half = self.field_um / 2.0
        return tile.x_um - half, tile.y_um - half

    @property
    def field_pixels(self) -> int:
        """The side of one field, in pixels: one tile's width."""
        return self[0].shape[-1]

    @property
    def field_um(self) -> float:
        """The side of one field, in µm."""
        return self.field_pixels * self.um_per_px

    @property
    def nbytes(self) -> int:
        extra = 0 if self.overview_planes is None else self.overview_planes.nbytes
        return super().nbytes + extra

    def save_to(self, root: Path) -> None:
        with self.lock:
            # The grid first, as a run writes it before any image.
            self.root = root
            self.write_grid()
            super().save_to(root)
            if self.overview_planes is not None:
                self.write_overview(root, self.overview_planes)

    def write_grid(self) -> None:
        if self.root is None:
            return
        path = self.root.with_name(f"{self.root.name}_{self.name}_grid.json")
        grid = {
            "um_per_px": self.um_per_px,
            "tiles": [asdict(tile) for tile in self.tiles],
            "placement": None if self.placement is None else asdict(self.placement),
        }
        path.write_text(json.dumps(grid, indent=2), encoding="utf-8")
        self.paths[GRID_KEY] = path

    def write_overview(self, root: Path, planes: np.ndarray) -> None:
        stem = root.with_name(f"{root.name}_{self.name}_overview")
        files = channel_paths(stem, self.channels)
        append_pages(files, planes)
        self.paths |= {f"{OVERVIEW_PREFIX}{label}": path for label, path in files.items()}

    @classmethod
    def load(cls: type[M], *, name: str, recording: Recording, files: dict[str, Path]) -> M:
        output = cls(name=name, recording=recording, root=None)
        grid = json.loads(files[GRID_KEY].read_text(encoding="utf-8"))
        output.um_per_px = float(grid["um_per_px"])
        output.tiles = tuple(
            MosaicTile(int(t["row"]), int(t["col"]), float(t["x_um"]), float(t["y_um"]))
            for t in grid["tiles"]
        )
        overview = {
            key.removeprefix(OVERVIEW_PREFIX): path
            for key, path in files.items()
            if key.startswith(OVERVIEW_PREFIX)
        }
        if overview:
            (output.overview_planes,) = read_pages(overview)
            output.placement = Placement(**grid["placement"])
        channels = {key: path for key, path in files.items() if cls.names_channel(key)}
        if channels:
            output.fill(read_pages(channels), list(channels))
        return output

    @classmethod
    def names_channel(cls, key: str) -> bool:
        """Whether ``key``, among a saved mosaic's files, names a channel of its tiles."""
        return key != GRID_KEY and not key.startswith(OVERVIEW_PREFIX)
