"""``NpzStack``: an ``ArrayStack`` saved as one ``.npz`` when its recording closes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from pyrpoc.core.recording import Recording

from .array_stack import ArrayStack

S = TypeVar("S", bound="NpzStack")


class NpzStack(ArrayStack):
    """``<root>_<name>.npz`` holding ``data``, every array stacked in order, and ``channels``."""

    # Buffered: an npz cannot be appended to, and everything is held in memory anyway.

    def close(self) -> None:
        if self.root is not None and self.arrays:
            path = self.root.with_name(f"{self.root.name}_{self.name}.npz")
            np.savez_compressed(
                str(path),
                data=np.stack(self.arrays, axis=0),
                channels=np.asarray(self.channels, dtype=str),
                **self.extra(),
            )
            self.paths = {self.name: path}
        super().close()

    def extra(self) -> dict[str, Any]:
        """More arrays to save beside the data, read back by ``take_extra``."""
        return {}

    def take_extra(self, archive: Any) -> None:
        del archive

    @classmethod
    def load(cls: type[S], *, name: str, recording: Recording, files: dict[str, Path]) -> S:
        output = cls(name=name, recording=recording, root=None)
        if files:
            (path,) = files.values()
            with np.load(str(path)) as archive:
                channels = [str(label) for label in archive["channels"]]
                output.fill(list(archive["data"]), channels)
                output.take_extra(archive)
        return output
