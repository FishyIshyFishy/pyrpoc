"""Buffered in memory, written once when the run ends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pyrpoc.structs.data_library.data import Cube3D, Samples4D, Spectrum1D
from pyrpoc.structs.data_library.dataset import Dataset
from pyrpoc.structs.data_library.saving import Writer, writer_registry


@writer_registry.register(Cube3D.name)
@writer_registry.register(Samples4D.name)
@writer_registry.register(Spectrum1D.name)
class NpzWriter(Writer):
    """``<root>_<output>.npz`` holding ``data`` and ``parameters``. The leading
    axis of ``data`` is one entry per published array, in publish order."""

    def __init__(self, root: Path, output: str, parameters: dict[str, Any]):
        super().__init__(root, output, parameters)
        self._buffer: list[np.ndarray] = []

    def write(self, dataset: Dataset, array: np.ndarray) -> None:
        self._buffer.append(np.asarray(array, dtype=np.float32))

    def finalize(self, dataset: Dataset, error: Exception | None) -> None:
        if not self._buffer:
            return
        path = self.root.with_name(f"{self.root.name}_{self.output}.npz")
        np.savez_compressed(
            str(path),
            data=np.stack(self._buffer, axis=0),
            parameters=np.asarray(self.parameters, dtype=object),
        )
        self.paths = {self.output: path}

    @classmethod
    def read(cls, files: dict[str, Path]) -> list[np.ndarray]:
        """One file per output; ``parameters`` is not read, since the recording's
        metadata file holds them, and reading it would unpickle."""
        (path,) = files.values()
        with np.load(str(path)) as archive:
            return list(archive["data"])
