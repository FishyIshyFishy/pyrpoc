"""``Scratch``: a loop's working copy of everything it did, kept in its folder until it finishes,
so a loop cut short can be carried on."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from pyrpoc.plugins.program_io.outputs import LoopEvent, event_from_dict, event_to_dict

SCRATCH_FILE = "loop_scratch.jsonl"


class Scratch:
    """``loop_scratch.jsonl`` in a folder: a line naming each recording that ran the loop, and a
    line for each of the loop's events."""

    def __init__(self, folder: Path):
        self.path = folder / SCRATCH_FILE

    def exists(self) -> bool:
        return self.path.exists()

    def recordings(self) -> list[str]:
        """The recordings that ran the loop so far, saved ones only."""
        return [line["recording"] for line in self.lines() if "recording" in line]

    def events(self) -> list[LoopEvent]:
        return [event_from_dict(line) for line in self.lines() if "event" in line]

    def lines(self) -> list[dict[str, Any]]:
        """Each line as written. A loop cut off mid-line leaves its last line partial, so that
        line is passed over."""
        texts = [text for text in self.path.read_text(encoding="utf-8").splitlines() if text]
        lines = [json.loads(text) for text in texts[:-1]]
        if texts:
            with contextlib.suppress(json.JSONDecodeError):
                lines.append(json.loads(texts[-1]))
        return lines

    def note_recording(self, name: str) -> None:
        self.write({"recording": name})

    def append(self, event: LoopEvent) -> None:
        self.write(event_to_dict(event))

    def write(self, line: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as scratch:
            scratch.write(json.dumps(line) + "\n")

    def discard(self) -> None:
        self.path.unlink()
