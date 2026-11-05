"""``PlacedImageOutput``: images that each cover a known rectangle of the dish."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

from pyrpoc.core.program_output import program_output_registry
from pyrpoc.core.recording import Recording

from .building_blocks.tiff import read_pages
from .image import ImageOutput

# The ``paths`` key of the placements, beside one key per channel.
PLACEMENTS_KEY = "placements"

P = TypeVar("P", bound="PlacedImageOutput")


@dataclass(frozen=True)
class Placement:
    """Where an image lies on the dish: its top-left pixel's position and its pixel size,
    in µm."""

    left_um: float
    top_um: float
    um_per_px: float


@program_output_registry.register("placed_image")
class PlacedImageOutput(ImageOutput):
    """An ``ImageOutput`` whose image ``i`` lies at ``placements[i]``, saved as
    ``<root>_<name>_placements.json``."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.placements: list[Placement] = []

    def add_placed(self, array: np.ndarray, channels: Sequence[str], placement: Placement) -> None:
        with self.lock:
            self.placements.append(placement)
            if self.root is not None:
                path = self.root.with_name(f"{self.root.name}_{self.name}_placements.json")
                path.write_text(json.dumps([asdict(each) for each in self.placements], indent=2))
                self.paths[PLACEMENTS_KEY] = path
        self.add(array, channels)

    def latest(self) -> tuple[np.ndarray, Placement]:
        """The newest image and where it lies."""
        with self.lock:
            if not self.arrays:
                raise ValueError(f"{self.name} holds no image")
            # By the image's index: its placement is stored just before it.
            index = len(self.arrays) - 1
            return self.arrays[index], self.placements[index]

    @classmethod
    def load(cls: type[P], *, name: str, recording: Recording, files: dict[str, Path]) -> P:
        output = cls(name=name, recording=recording, root=None)
        channels = {key: path for key, path in files.items() if key != PLACEMENTS_KEY}
        if channels:
            output.fill(read_pages(channels), list(channels))
        raw = json.loads(files[PLACEMENTS_KEY].read_text(encoding="utf-8"))
        output.placements = [
            Placement(float(p["left_um"]), float(p["top_um"]), float(p["um_per_px"])) for p in raw
        ]
        return output
