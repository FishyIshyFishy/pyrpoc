"""Channel-per-file TIFFs: one appended float32 page per array added."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tifffile


def channel_paths(root: Path, channels: Sequence[str]) -> dict[str, Path]:
    """``<root>_<channel>.tiff`` for each channel, emptied."""
    paths = {label: root.with_name(f"{root.name}_{label}.tiff") for label in channels}
    for path in paths.values():
        path.unlink(missing_ok=True)
    return paths


def append_pages(paths: dict[str, Path], array: np.ndarray) -> None:
    """One page per channel of a ``(C, H, W)`` array, onto that channel's file."""
    for path, plane in zip(paths.values(), array, strict=True):
        with tifffile.TiffWriter(str(path), append=True) as writer:
            writer.write(plane)


def read_pages(files: dict[str, Path]) -> list[np.ndarray]:
    """Every page as ``(C, H, W)`` arrays, channels in the order of ``files``."""
    # Each array appended its own page, so pages are read one by one rather than as a series.
    per_channel = []
    for path in files.values():
        with tifffile.TiffFile(str(path)) as tif:
            per_channel.append([page.asarray() for page in tif.pages])
    counts = {len(pages) for pages in per_channel}
    if len(counts) != 1:
        raise ValueError(f"channel files hold different page counts: {sorted(counts)}")
    return [np.stack(planes) for planes in zip(*per_channel, strict=True)]
