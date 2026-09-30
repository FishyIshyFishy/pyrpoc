"""One appended TIFF per channel, one page per publish."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from pyrpoc.structs.data_library.data import Image2D
from pyrpoc.structs.data_library.dataset import Dataset
from pyrpoc.structs.data_library.saving import Writer, writer_registry


@writer_registry.register(Image2D.name)
class TiffWriter(Writer):
    """``<root>_<channel>.tiff``, appended float32, readable while the run goes."""

    def write(self, dataset: Dataset, array: np.ndarray) -> None:
        channels = [array[index] for index in range(array.shape[0])]

        if not self.paths:
            labels = dataset.resolved_channel_labels(len(channels))
            root = self.root
            self.paths = {label: root.with_name(f"{root.name}_{label}.tiff") for label in labels}
            for path in self.paths.values():
                path.unlink(missing_ok=True)

        if len(channels) != len(self.paths):
            raise ValueError("channel count does not match the configured save layout")

        for path, channel_plane in zip(self.paths.values(), channels, strict=True):
            with tifffile.TiffWriter(str(path), append=True) as writer:
                writer.write(np.asarray(channel_plane, dtype=np.float32))

    @classmethod
    def read(cls, files: dict[str, Path]) -> list[np.ndarray]:
        """``(C, H, W)`` frames, a channel per file. Every publish appended its
        own page, so pages are read one by one rather than as one series."""
        per_channel = []
        for path in files.values():
            with tifffile.TiffFile(str(path)) as tif:
                per_channel.append([page.asarray() for page in tif.pages])
        counts = {len(pages) for pages in per_channel}
        if len(counts) != 1:
            raise ValueError(f"channel files hold different frame counts: {sorted(counts)}")
        return [np.stack(planes) for planes in zip(*per_channel, strict=True)]
