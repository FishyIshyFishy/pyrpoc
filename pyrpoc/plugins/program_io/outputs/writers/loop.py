"""A loop's log, one JSON line per event appended as it happens, and each unit's frames as
TIFF stacks of their own."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.saving import Writer, writer_registry
from pyrpoc.plugins.program_io.outputs.arrays import Image
from pyrpoc.plugins.program_io.outputs.loop import LoopEvent, UnitFrame

from .tiff import append_pages, read_pages


@writer_registry.register(LoopEvent.name)
class EventLogWriter(Writer[LoopEvent]):
    """``<root>_<output>.jsonl``, a line appended and flushed for each event."""

    def write(self, program_output: ProgramOutput[LoopEvent], value: LoopEvent) -> None:
        path = self.root.with_name(f"{self.root.name}_{self.output}.jsonl")
        with path.open("a", encoding="utf-8") as log:
            log.write(value.to_json_line() + "\n")
        self.paths = {self.output: path}

    @classmethod
    def read(cls, kind: type[LoopEvent], files: dict[str, Path]) -> LoopEvent:
        (path,) = files.values()
        raw = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        return kind(
            float(raw["time_s"]),
            str(raw["event"]),
            raw["unit"],
            str(raw["routine"]),
            raw["details"],
        )


@writer_registry.register(UnitFrame.name)
class UnitFramesWriter(Writer[UnitFrame]):
    """``<root>_<output>_unit<id>_<routine>_<channel>.tiff``, a page appended per frame."""

    def __init__(self, root: Path, output: str, parameters: dict[str, Any]):
        super().__init__(root, output, parameters)
        self.stacks: dict[tuple[int, str], dict[str, Path]] = {}

    def write(self, program_output: ProgramOutput[UnitFrame], value: UnitFrame) -> None:
        key = (value.unit, value.routine)
        if key not in self.stacks:
            stem = f"{self.root.name}_{self.output}_unit{value.unit}_{value.routine}"
            files = {
                label: self.root.with_name(f"{stem}_{label}.tiff") for label in value.image.channels
            }
            for path in files.values():
                path.unlink(missing_ok=True)
            self.stacks[key] = files
            self.paths |= {
                f"{value.unit}:{value.routine}:{label}": path for label, path in files.items()
            }
        append_pages(self.stacks[key], value.image)

    @classmethod
    def read(cls, kind: type[UnitFrame], files: dict[str, Path]) -> UnitFrame:
        """The last page of the first stack: a loop is read back through its log."""
        unit, routine, _ = next(iter(files)).split(":", 2)
        stack = {
            key.split(":", 2)[2]: path
            for key, path in files.items()
            if key.startswith(f"{unit}:{routine}:")
        }
        pages = read_pages(stack)
        last: Image = pages[-1]
        return kind(last, int(unit), routine, len(pages) - 1, 0.0, 0.0)
