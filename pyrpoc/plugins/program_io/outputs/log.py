"""``LogOutput``: what a run did, one event at a time, in order, written as it happens. Each kind
of log is a subclass naming its own events."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import Field, asdict
from pathlib import Path
from typing import Any, ClassVar, Generic, Protocol, TypeVar

from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.recording import Recording


class LogEvent(Protocol):
    """One thing a run did: its kind, when on the instrument's clock, and a line saying what."""

    # Every event is a flat dataclass, so it is written as its fields.
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]
    kind: ClassVar[str]

    @property
    def time_s(self) -> float: ...

    def text(self) -> str: ...


E = TypeVar("E", bound=LogEvent)
L = TypeVar("L", bound="LogOutput[Any]")


def event_to_dict(event: LogEvent) -> dict[str, Any]:
    return {"event": event.kind, **asdict(event)}


def event_from_dict(raw: dict[str, Any], kinds: Mapping[str, type[E]]) -> E:
    """The event of one of ``kinds`` written as ``raw``, its lists back as tuples."""
    fields = {key: as_tuples(value) for key, value in raw.items() if key != "event"}
    return kinds[raw["event"]](**fields)


def as_tuples(value: Any) -> Any:
    return tuple(as_tuples(each) for each in value) if isinstance(value, list) else value


class LogOutput(ProgramOutput, Generic[E]):
    """Events, each appended to ``<root>_<name>.jsonl`` as it happens, so a run cut short still
    says everything it did up to then."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.events: list[E] = []
        self.nbytes_logged = 0

    def kinds(self) -> Mapping[str, type[E]]:
        """Each kind of event this log holds, by its ``kind``."""
        raise NotImplementedError

    def log(self, event: E) -> None:
        self.log_all([event])

    def log_all(self, events: Sequence[E]) -> None:
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

    def since(self, start: int) -> list[E]:
        """The events from the ``start``-th on, oldest first."""
        with self.lock:
            return self.events[start:]

    @property
    def nbytes(self) -> int:
        return self.nbytes_logged

    @classmethod
    def load(cls: type[L], *, name: str, recording: Recording, files: dict[str, Path]) -> L:
        output = cls(name=name, recording=recording, root=None)
        if files:
            (path,) = files.values()
            kinds = output.kinds()
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    output.events.append(event_from_dict(json.loads(line), kinds))
                    output.nbytes_logged += len(line)
        return output
