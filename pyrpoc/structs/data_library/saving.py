"""How data reaches disk: where an acquisition saves, and the file formats
that write and read each kind of data. Formats register in ``writer_registry``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pyrpoc.structs.plugins.params import ParameterError
from pyrpoc.structs.registry import Registry

if TYPE_CHECKING:  # pragma: no cover
    from .dataset import Dataset


@dataclass
class SaveTarget:
    """What an acquisition is called, where it goes, and whether it goes.

    Its own argument to the executor rather than a parameter block, because
    where bytes land does not depend on which program produced them. ``name``
    is a bare filename and also names the run in the data panel with saving off.
    """

    name: str = "acquisition"
    directory: str = ""
    enabled: bool = False

    @property
    def filename(self) -> str:
        """``name`` with no directory and no TIFF suffix; the writers append
        their own ``_<channel>.tiff``."""
        stem = Path(self.name.strip()).name
        if stem.lower().endswith((".tif", ".tiff")):
            stem = stem.rsplit(".", 1)[0]
        return stem

    @property
    def folder(self) -> Path:
        """Where files go. No directory means the working directory."""
        text = self.directory.strip()
        return Path(text).expanduser() if text else Path.cwd()

    @property
    def root(self) -> Path:
        """The base path the writers hang their suffixes off."""
        if not self.filename:
            raise ParameterError("Name is required when saving is enabled")
        return self.folder / self.filename


class Writer:
    """Puts one output's arrays on disk as they arrive, and reads them back:
    one file format, registered in ``writer_registry`` for each kind of
    ``Data`` it saves.

    ``root`` is the base path every file of the run hangs its suffix off, and
    ``parameters`` the run's encoded blocks, for formats that carry them.
    """

    def __init__(self, root: Path, output: str, parameters: dict[str, Any]):
        self.root = root
        self.output = output
        self.parameters = parameters
        # What was written, keyed as ``read`` expects them back.
        self.paths: dict[str, Path] = {}

    def write(self, dataset: Dataset, array: np.ndarray) -> None:
        raise NotImplementedError

    def finalize(self, dataset: Dataset, error: Exception | None) -> None:
        del dataset, error

    @classmethod
    def read(cls, files: dict[str, Path]) -> list[np.ndarray]:
        """Every frame ``paths`` named, in publish order. ``files`` has the same
        keys ``paths`` had when the recording was written."""
        raise NotImplementedError


# Keyed by the name of the kind of ``Data`` saved; one writer may take several.
writer_registry: Registry[Writer] = Registry("WriterRegistry", Writer, stamp=None)
