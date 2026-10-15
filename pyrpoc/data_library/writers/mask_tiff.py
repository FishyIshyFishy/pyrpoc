"""One appended uint8 TIFF per output, one page per publish."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from pyrpoc.structs.data_library.data import Mask2D
from pyrpoc.structs.data_library.dataset import Dataset
from pyrpoc.structs.data_library.saving import Writer, writer_registry


@writer_registry.register(Mask2D.name)
class MaskTiffWriter(Writer):
    """``<root>_<output>.tiff``, appended uint8, so a mask reloads as the
    decision per pixel it was rather than a float image."""

    def write(self, dataset: Dataset, array: np.ndarray) -> None:
        del dataset
        if not self.paths:
            path = self.root.with_name(f"{self.root.name}_{self.output}.tiff")
            path.unlink(missing_ok=True)
            self.paths = {self.output: path}
        with tifffile.TiffWriter(str(self.paths[self.output]), append=True) as writer:
            writer.write(np.asarray(array, dtype=np.uint8))

    @classmethod
    def read(cls, files: dict[str, Path]) -> list[np.ndarray]:
        (path,) = files.values()
        with tifffile.TiffFile(str(path)) as tif:
            return [page.asarray() for page in tif.pages]
