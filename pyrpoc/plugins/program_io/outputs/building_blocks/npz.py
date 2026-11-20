"""``NpzStack``: an ``ArrayStack`` saved as one ``.npz`` when its recording closes."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import numpy as np

from pyrpoc.core.recording import Recording
from pyrpoc.utils.file_formats.npz import read_npz, write_npz

from .array_stack import ArrayStack

S = TypeVar("S", bound="NpzStack")


class NpzStack(ArrayStack):
    """``<root>_<name>.npz`` holding ``data``, every array stacked in order, and ``channels``."""

    # Buffered: an npz cannot be appended to, and everything is held in memory anyway.

    def close(self) -> None:
        if self.root is not None and self.arrays:
            self.write_file(self.root)
        super().close()

    def save_to(self, root: Path) -> None:
        with self.lock:
            self.root = root
            if self.arrays:
                self.write_file(root)

    def write_file(self, root: Path) -> None:
        path = root.with_name(f"{root.name}_{self.name}.npz")
        write_npz(
            path,
            {
                "data": np.stack(self.arrays, axis=0),
                "channels": np.asarray(self.channels, dtype=str),
                **self.extra(),
            },
        )
        self.paths = {self.name: path}

    def extra(self) -> dict[str, np.ndarray]:
        """More arrays to save beside the data, read back by ``take_extra``."""
        return {}

    def take_extra(self, arrays: dict[str, np.ndarray]) -> None:
        del arrays

    @classmethod
    def load(cls: type[S], *, name: str, recording: Recording, files: dict[str, Path]) -> S:
        output = cls(name=name, recording=recording, root=None)
        if files:
            (path,) = files.values()
            arrays = read_npz(path)
            channels = [str(label) for label in arrays["channels"]]
            output.fill(list(arrays["data"]), channels)
            output.take_extra(arrays)
        return output
