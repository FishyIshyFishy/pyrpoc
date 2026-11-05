"""A small result as one JSON file, rewritten on each publish."""

from __future__ import annotations

import json
from pathlib import Path

from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.saving import Writer, writer_registry
from pyrpoc.plugins.program_io.outputs.records import GalvoCalibration, Record


@writer_registry.register(GalvoCalibration.name)
class JsonWriter(Writer[Record]):
    """``<root>_<output>.json``, holding the latest value."""

    def write(self, program_output: ProgramOutput[Record], value: Record) -> None:
        path = self.root.with_name(f"{self.root.name}_{self.output}.json")
        path.write_text(json.dumps(value.to_json(), indent=2), encoding="utf-8")
        self.paths = {self.output: path}

    @classmethod
    def read(cls, kind: type[Record], files: dict[str, Path]) -> Record:
        (path,) = files.values()
        return kind.from_json(json.loads(path.read_text(encoding="utf-8")))
