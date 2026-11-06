"""``LoopLogOutput``: everything a loop did, in order, written as it happens."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pyrpoc.core.program_output import ProgramOutput, program_output_registry
from pyrpoc.core.recording import Recording

from .loop_events import LoopEvent, event_from_dict, event_to_dict
from .loop_view import LoopFold, LoopView


@program_output_registry.register("loop_log")
class LoopLogOutput(ProgramOutput):
    """A loop's events, each appended to ``<root>_<name>.jsonl`` as it happens, so a loop cut
    short still says everything it did up to then."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.events: list[LoopEvent] = []
        self.nbytes_logged = 0

    def log(self, event: LoopEvent) -> None:
        self.log_all([event])

    def log_all(self, events: Sequence[LoopEvent]) -> None:
        lines = [json.dumps(event_to_dict(event)) for event in events]
        with self.lock:
            self.events.extend(events)
            self.nbytes_logged += sum(len(line) for line in lines)
            if self.root is not None:
                path = self.root.with_name(f"{self.root.name}_{self.name}.jsonl")
                with path.open("a", encoding="utf-8") as log:
                    log.writelines(line + "\n" for line in lines)
                self.paths = {self.name: path}
        self.changed()

    def view(self) -> LoopView:
        """What the log says the loop is doing as of its latest event."""
        with self.lock:
            events = list(self.events)
        fold = LoopFold()
        for event in events:
            fold.take(event)
        return fold.view()

    @property
    def nbytes(self) -> int:
        return self.nbytes_logged

    @classmethod
    def load(cls, *, name: str, recording: Recording, files: dict[str, Path]) -> LoopLogOutput:
        output = cls(name=name, recording=recording, root=None)
        if files:
            (path,) = files.values()
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    output.events.append(event_from_dict(json.loads(line)))
                    output.nbytes_logged += len(line)
        return output
