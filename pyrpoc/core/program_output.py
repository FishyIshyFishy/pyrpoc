"""``ProgramOutput``: one named part of a recording, of one type that owns its content and its
files, and ``program_output_registry``, every type by name."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from .registry import Registry


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, kw_only=True)
class Provenance:
    """What produced this output: the program, when and under what name it ran, and its
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


OutputT = TypeVar("OutputT", bound="ProgramOutput")


class ProgramOutput:
    """One named part of a recording. Each type is a subclass that holds its own content, says
    how programs add to it and panels read it, and writes it to files and reads it back."""

    # Set by ``program_output_registry``; a recording on disk names its outputs' types by it.
    type_key: str = "program_output"

    def __init__(self, *, name: str, provenance: Provenance, origin: Origin, root: Path | None):
        self.id = f"{name}-{uuid4().hex[:12]}"
        self.name = name
        self.provenance = provenance
        self.origin = origin
        # Where its files go, beside the recording's; None when it is not saved.
        self.root = root
        # What it wrote, keyed as ``load`` expects them back.
        self.paths: dict[str, Path] = {}
        # Nothing more will be added. Only an acquisition is still being written.
        self.finished = origin is not Origin.ACQUIRED
        # The recording's metadata file, once it is on disk.
        self.meta_path: Path | None = None
        # Content is added on a run's thread and read on the GUI thread.
        self.lock = threading.RLock()
        self._subscribers: list[Callable[[ProgramOutput], None]] = []

    @property
    def nbytes(self) -> int:
        """Memory held, in bytes."""
        raise NotImplementedError

    @classmethod
    def load(
        cls: type[OutputT], *, name: str, provenance: Provenance, files: dict[str, Path]
    ) -> OutputT:
        """The output written to ``files``, keyed as ``paths`` was, finished."""
        raise NotImplementedError

    def close(self, error: Exception | None) -> None:
        """Finish its files: nothing more will be added."""
        del error
        self.finished = True

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

    def subscribe(self, callback: Callable[[ProgramOutput], None]) -> None:
        with self.lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[ProgramOutput], None]) -> None:
        with self.lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def changed(self) -> None:
        """Tell every subscriber its content changed."""
        # Called on the thread that changed it, so subscribers must not touch Qt.
        with self.lock:
            listeners = list(self._subscribers)
        for callback in listeners:
            callback(self)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name}>"


# Every type of program output, by ``type_key``, so a recording on disk finds its classes again.
program_output_registry: Registry[ProgramOutput] = Registry(
    "ProgramOutputRegistry", ProgramOutput, stamp="type_key"
)
