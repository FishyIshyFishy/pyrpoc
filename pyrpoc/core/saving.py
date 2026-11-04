"""Where a run saves (``SaveTarget``), and the file formats that write and
read each kind of data (``Writer`` and ``writer_registry``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pyrpoc.core.program_input import ParameterError
from pyrpoc.core.program_output import Data
from pyrpoc.core.registry import Registry

if TYPE_CHECKING:  # pragma: no cover
    from .program_output import ProgramOutput

D = TypeVar("D", bound=Data)


@dataclass
class SaveTarget:
    """What a run is called, where it saves, and whether saving is on."""

    # Not a program input: where files land does not depend on the program.
    # A bare filename, which also names the run when saving is off.
    name: str = "acquisition"
    directory: str = ""
    enabled: bool = False

    @property
    def filename(self) -> str:
        """``name`` with no directory and no TIFF suffix."""
        stem = Path(self.name.strip()).name
        if stem.lower().endswith((".tif", ".tiff")):
            stem = stem.rsplit(".", 1)[0]
        return stem

    @property
    def folder(self) -> Path | None:
        """Where files go, or None when no folder is set."""
        text = self.directory.strip()
        return Path(text).expanduser() if text else None

    @property
    def root(self) -> Path:
        """The base path every file of the run hangs its suffix off."""
        folder = self.folder
        if folder is None:
            raise ParameterError("A folder to save into is required when saving is enabled")
        if not self.filename:
            raise ParameterError("Name is required when saving is enabled")
        return folder / self.filename


class Writer(Generic[D]):
    """One file format: writes one output's values to disk as they are
    published, and reads the last one back."""

    def __init__(self, root: Path, output: str, parameters: dict[str, Any]):
        # The base path every file of the run hangs its suffix off.
        self.root = root
        self.output = output
        # The run's encoded program inputs, for formats that carry them.
        self.parameters = parameters
        # What was written, keyed as ``read`` expects them back.
        self.paths: dict[str, Path] = {}

    def write(self, program_output: ProgramOutput[D], value: D) -> None:
        raise NotImplementedError

    def finalize(self, program_output: ProgramOutput[D], error: Exception | None) -> None:
        del program_output, error

    @classmethod
    def read(cls, kind: type[D], files: dict[str, Path]) -> D:
        """The last value written to ``files``, keyed as ``paths`` was."""
        raise NotImplementedError


# Keyed by the name of the kind of ``Data`` saved; one writer may take several.
writer_registry: Registry[Writer[Any]] = Registry("WriterRegistry", Writer, stamp=None)
