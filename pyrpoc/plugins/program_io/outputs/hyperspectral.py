"""``HyperspectralOutput``: images taken one per delay of a delay sweep, each with the delay stage
position it was taken at."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pyrpoc.core.program_output import program_output_registry
from pyrpoc.core.recording import Recording

from .building_blocks.tiff import read_pages
from .image import ImageOutput

# The ``paths`` key of the delays, beside one key per channel.
DELAYS_KEY = "delays"


@program_output_registry.register("hyperspectral")
class HyperspectralOutput(ImageOutput):
    """An ``ImageOutput`` whose image ``i`` was taken with the delay stage at ``delays_um[i]``.
    The delays are saved as ``<root>_<name>_delays.json`` after every image, so a sweep cut
    short still says where each image was taken."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.delays_um: list[float] = []

    def add_at(self, delay_um: float, array: np.ndarray, channels: Sequence[str]) -> None:
        """Add the ``(C, H, W)`` image taken with the delay stage at ``delay_um``."""
        # The delay is kept before the image, so a reader never finds an image without one.
        with self.lock:
            self.delays_um.append(delay_um)
            self.write_delays()
        self.add(array, channels)

    def taken(self) -> tuple[tuple[float, ...], tuple[np.ndarray, ...]]:
        """Every image so far and the delay of each, paired."""
        with self.lock:
            count = len(self.arrays)
            return tuple(self.delays_um[:count]), tuple(self.arrays)

    def write_delays(self) -> None:
        if self.root is None:
            return
        path = self.root.with_name(f"{self.root.name}_{self.name}_delays.json")
        path.write_text(json.dumps({"delays_um": self.delays_um}, indent=2), encoding="utf-8")
        self.paths[DELAYS_KEY] = path

    @classmethod
    def load(
        cls, *, name: str, recording: Recording, files: dict[str, Path]
    ) -> HyperspectralOutput:
        output = cls(name=name, recording=recording, root=None)
        channels = {key: path for key, path in files.items() if key != DELAYS_KEY}
        if channels:
            output.fill(read_pages(channels), list(channels))
        if DELAYS_KEY in files:
            raw = json.loads(files[DELAYS_KEY].read_text(encoding="utf-8"))
            output.delays_um = [float(each) for each in raw["delays_um"]]
        return output
