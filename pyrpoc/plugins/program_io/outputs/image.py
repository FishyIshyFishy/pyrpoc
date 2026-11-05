"""``ImageOutput``: pictures added one after another, one plane per channel."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyrpoc.core.program_output import program_output_registry
from pyrpoc.core.recording import Recording

from .building_blocks.array_stack import ArrayStack
from .building_blocks.tiff import append_pages, channel_paths, read_pages


@program_output_registry.register("image")
class ImageOutput(ArrayStack):
    """``(channel, y, x)`` images, saved as ``<root>_<channel>.tiff`` with one page per image,
    readable while the run goes."""

    axes = ("channel", "y", "x")

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        # Apart from ``paths``, which a daughter type may add its own files to.
        self.channel_files: dict[str, Path] = {}

    def write(self, root: Path, array: np.ndarray) -> None:
        if not self.channel_files:
            self.channel_files = channel_paths(root, self.channels)
            self.paths |= self.channel_files
        append_pages(self.channel_files, array)

    @classmethod
    def load(cls, *, name: str, recording: Recording, files: dict[str, Path]) -> ImageOutput:
        output = cls(name=name, recording=recording, root=None)
        if files:
            output.fill(read_pages(files), list(files))
        return output
