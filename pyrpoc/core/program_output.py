"""``ProgramOutput``: one named output of a run, whose values are ``Data`` of one kind: its
latest value, where it came from, and who is told when it changes."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar
from uuid import uuid4

from .registry import Registry

if TYPE_CHECKING:  # pragma: no cover
    from .saving import Writer


class Data:
    """One value of data. ``name`` is the class name, which a recording's
    ``kind`` field holds."""

    name: ClassVar[str] = "Data"

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.name = cls.__name__

    @property
    def nbytes(self) -> int:
        """Memory held, in bytes."""
        raise NotImplementedError

    def summary(self) -> str:
        """One line saying what this is."""
        raise NotImplementedError


# Every kind of ``Data``, by ``name``, so a recording's kind string finds its class.
data_kind_registry: Registry[Data] = Registry("DataKindRegistry", Data, stamp=None)

D = TypeVar("D", bound=Data)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, kw_only=True)
class Provenance:
    """What produced this data: the program, when and under what name it ran, and its
    inputs and devices."""

    program_key: str
    started_at: str
    # What the run was called: the save filename, set whether or not it was
    # saved. Empty for an unnamed run, which then goes by its program.
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    devices: dict[str, Any] = field(default_factory=dict)
    run_id: int = 0


class Origin(Enum):
    """Where a program output came from."""

    ACQUIRED = "acquired"
    LOADED = "loaded"


class ProgramOutput(Generic[D]):
    """One named output of one run: the latest value of one ``kind`` in memory,
    with every published value passed to its writer."""

    def __init__(
        self,
        *,
        name: str,
        kind: type[D],
        provenance: Provenance,
        origin: Origin,
        writer: Writer[D] | None,
    ):
        self.id = f"{name}-{uuid4().hex[:12]}"
        self.name = name
        self.kind = kind
        self.provenance = provenance
        self.origin = origin
        # No more values will arrive. Only an acquisition is still being written.
        self.finished = origin is not Origin.ACQUIRED
        # The recording's metadata file, once it is on disk.
        self.meta_path: Path | None = None
        self.notes = ""
        self.metadata: dict[str, Any] = {}
        self.writer = writer
        self.value: D | None = None
        # How many values were published into it; every one reached the writer.
        self.published = 0

        self._subscribers: list[Callable[[ProgramOutput[D]], None]] = []
        self._lock = threading.RLock()

    @property
    def run_id(self) -> int:
        return self.provenance.run_id

    @property
    def run_name(self) -> str:
        """What the run was called, or its program when it was not named."""
        return self.provenance.name or self.provenance.program_key

    @property
    def started_time(self) -> str:
        """Local time of day the run started."""
        # No date: a session lasts a day, so a date column would repeat on every row.
        started = datetime.fromisoformat(self.provenance.started_at)
        return started.astimezone().strftime("%H:%M:%S")

    @property
    def label(self) -> str:
        """One line naming this program output."""
        return " · ".join((self.started_time, self.run_name, self.name))

    @property
    def nbytes(self) -> int:
        value = self.value
        return 0 if value is None else value.nbytes

    def publish(self, value: D) -> None:
        """Replace the value, save it, then notify subscribers."""
        # Runs on the worker thread, so subscribers must not touch Qt.
        with self._lock:
            self.value = value
            self.published += 1
        if self.writer is not None:
            self.writer.write(self, value)
        self.notify()

    def subscribe(self, callback: Callable[[ProgramOutput[D]], None]) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[ProgramOutput[D]], None]) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def notify(self) -> None:
        with self._lock:
            listeners = list(self._subscribers)
        for callback in listeners:
            callback(self)

    def finalize(self, error: Exception | None) -> None:
        self.finished = True
        if self.writer is not None:
            self.writer.finalize(self, error)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ProgramOutput {self.name} {self.kind.name} published={self.published}>"
