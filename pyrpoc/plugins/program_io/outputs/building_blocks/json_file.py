"""``JsonFile``: a program output whose content is small, saved whole as one JSON file each time
it changes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.recording import Recording
from pyrpoc.utils.file_formats.json_file import read_json, write_json

J = TypeVar("J", bound="JsonFile")


class JsonFile(ProgramOutput):
    """``<root>_<name>.json``, rewritten by ``save`` and read back through ``take``."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.nbytes_saved = 0

    def save(self, content: dict[str, Any]) -> None:
        """Write ``content`` as the file, when saving."""
        self.nbytes_saved = len(json.dumps(content, indent=2))
        if self.root is None:
            return
        path = self.root.with_name(f"{self.root.name}_{self.name}.json")
        write_json(path, content)
        self.paths = {self.name: path}

    def take(self, raw: dict[str, Any]) -> None:
        """Set the content from the file's JSON."""
        raise NotImplementedError

    @property
    def nbytes(self) -> int:
        return self.nbytes_saved

    @classmethod
    def load(cls: type[J], *, name: str, recording: Recording, files: dict[str, Path]) -> J:
        output = cls(name=name, recording=recording, root=None)
        if files:
            (path,) = files.values()
            output.nbytes_saved = path.stat().st_size
            output.take(read_json(path))
        return output
