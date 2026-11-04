"""One appended TIFF per channel, one page per publish."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.saving import Writer, writer_registry
from pyrpoc.plugins.program_io.outputs.arrays import Image


def channel_paths(root: Path, image: Image) -> dict[str, Path]:
    """``<root>_<channel>.tiff`` for each of ``image``'s channels, emptied."""
    paths = {label: root.with_name(f"{root.name}_{label}.tiff") for label in image.channels}
    for path in paths.values():
        path.unlink(missing_ok=True)
    return paths


def append_pages(paths: dict[str, Path], image: Image) -> None:
    """One float32 page per channel, onto that channel's file."""
    if list(paths) != list(image.channels):
        raise ValueError("an image's channels do not match the files it is saved in")
    for path, plane in zip(paths.values(), image.array, strict=True):
        with tifffile.TiffWriter(str(path), append=True) as writer:
            writer.write(plane)


def read_pages(files: dict[str, Path]) -> list[Image]:
    """Every page, as ``Image``s labelled by the files' keys."""
    # Each publish appended its own page, so pages are read one by one rather than as a series.
    per_channel = []
    for path in files.values():
        with tifffile.TiffFile(str(path)) as tif:
            per_channel.append([page.asarray() for page in tif.pages])
    counts = {len(pages) for pages in per_channel}
    if len(counts) != 1:
        raise ValueError(f"channel files hold different page counts: {sorted(counts)}")
    labels = list(files)
    return [Image(np.stack(planes), labels) for planes in zip(*per_channel, strict=True)]


@writer_registry.register(Image.name)
class TiffWriter(Writer[Image]):
    """``<root>_<channel>.tiff``, appended float32, readable while the run goes."""

    def write(self, program_output: ProgramOutput[Image], value: Image) -> None:
        if not self.paths:
            self.paths = channel_paths(self.root, value)
        append_pages(self.paths, value)

    @classmethod
    def read(cls, kind: type[Image], files: dict[str, Path]) -> Image:
        return read_pages(files)[-1]
