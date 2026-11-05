"""``SaveTarget``: what a run is called and where it saves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyrpoc.core.program_input import ParameterError


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
