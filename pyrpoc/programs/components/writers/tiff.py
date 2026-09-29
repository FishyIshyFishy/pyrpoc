"""One appended TIFF per channel, one page per publish."""

from __future__ import annotations

import numpy as np
import tifffile

from pyrpoc.structs.data import Dataset, Image2D, Writer
from pyrpoc.structs.registries import writer_registry


@writer_registry.register(Image2D.name)
class TiffWriter(Writer):
    """``<root>_<channel>.tiff``, appended float32, readable while the run goes."""

    metadata_key = "tiff_paths"

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
