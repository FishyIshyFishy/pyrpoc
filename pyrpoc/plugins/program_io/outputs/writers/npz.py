"""Buffered in memory, written once when the run ends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.saving import Writer, writer_registry
from pyrpoc.plugins.program_io.outputs.arrays import ArrayData, DecayHistogram, RawSamples, Spectrum


@writer_registry.register(DecayHistogram.name)
@writer_registry.register(RawSamples.name)
@writer_registry.register(Spectrum.name)
class NpzWriter(Writer[ArrayData]):
    """``<root>_<output>.npz`` holding ``data``, ``channels`` and ``parameters``.
    The leading axis of ``data`` is one entry per publish, in publish order."""

    def __init__(self, root: Path, output: str, parameters: dict[str, Any]):
        super().__init__(root, output, parameters)
        self._buffer: list[np.ndarray] = []
        self._channels: tuple[str, ...] = ()

    def write(self, program_output: ProgramOutput[ArrayData], value: ArrayData) -> None:
        self._buffer.append(value.array)
        self._channels = tuple(value.channels)

    def finalize(self, program_output: ProgramOutput[ArrayData], error: Exception | None) -> None:
        if not self._buffer:
            return
        path = self.root.with_name(f"{self.root.name}_{self.output}.npz")
        np.savez_compressed(
            str(path),
            data=np.stack(self._buffer, axis=0),
            channels=np.asarray(self._channels, dtype=str),
            parameters=np.asarray(self.parameters, dtype=object),
        )
        self.paths = {self.output: path}

    @classmethod
    def read(cls, kind: type[ArrayData], files: dict[str, Path]) -> ArrayData:
        """The last entry."""
        # ``parameters`` is not read: the recording's metadata file holds them, and reading it
        # would unpickle.
        (path,) = files.values()
        with np.load(str(path)) as archive:
            return kind(archive["data"][-1], [str(label) for label in archive["channels"]])
