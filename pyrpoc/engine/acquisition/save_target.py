"""``SaveTarget``: what a run is called and where it saves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
        """The base path every file of the run hangs its suffix off, for a target with a
        folder."""
        folder = self.folder
        # Narrows for the type checker: a run saves only once its readiness check found one.
        if folder is None:
            raise ValueError("a save target with no folder has no root")
        return folder / self.filename
