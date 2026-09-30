"""A dataset: one output's frames, where they came from, and who is told
when a frame arrives. Programs publish into one; the data library, data panels
and acquisition read it."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from .data import Data

if TYPE_CHECKING:  # pragma: no cover
    from .saving import Writer


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, kw_only=True)
class Provenance:
    """What produced this data. Written into the saved metadata."""

    program_key: str
    started_at: str
    # What the run was called: the save filename, set whether or not it was
    # saved. Empty for an unnamed run, which then goes by its program.
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    devices: dict[str, Any] = field(default_factory=dict)
    run_id: int = 0


class Origin(Enum):
    """Where a dataset came from, which decides what may be done with it."""

    ACQUIRED = "acquired"
    LOADED = "loaded"
    # Drawn by the user, like a mask: work that exists nowhere else.
    AUTHORED = "authored"


class Dataset:
    """One named output of one recording: frames of one kind of ``Data``."""

    def __init__(
        self,
        *,
        output: str,
        spec: type[Data],
        provenance: Provenance,
        origin: Origin,
        writer: Writer | None = None,
    ):
        self.id = f"{output}-{uuid4().hex[:12]}"
        self.output = output
        self.spec = spec
        self.provenance = provenance
        self.origin = origin
        # No more frames will arrive. Only an acquisition is still being written.
        self.finished = origin is not Origin.ACQUIRED
        # The recording's metadata file, once it is on disk.
        self.meta_path: Path | None = None
        self.notes = ""
        self.channel_labels: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.writer = writer

        self._frames: list[np.ndarray] = []
        self._nbytes = 0
        self._subscribers: list[Callable[[Dataset], None]] = []
        self._lock = threading.RLock()

    @property
    def run_id(self) -> int:
        return self.provenance.run_id

    @property
    def name(self) -> str:
        return self.provenance.name or self.provenance.program_key

    @property
    def started_time(self) -> str:
        """Local time of day the run started. No date: a session lasts a day,
        so a date column would repeat on every row."""
        started = datetime.fromisoformat(self.provenance.started_at)
        return started.astimezone().strftime("%H:%M:%S")

    @property
    def label(self) -> str:
        """One line naming this dataset, for pickers that have no columns."""
        return " · ".join((self.started_time, self.name, self.output))

    def resolved_channel_labels(self, count: int) -> list[str]:
        if len(self.channel_labels) == count:
            return list(self.channel_labels)
        return [f"channel_{index}" for index in range(count)]

    def append(self, array: np.ndarray) -> None:
        """Validate, store, save, then notify. Runs on the worker thread, so
        subscribers must not touch Qt; the data library's model re-emits for them."""
        frame = self.spec.coerce(array)
        with self._lock:
            self._frames.append(frame)
            self._nbytes += frame.nbytes
            if not self.channel_labels and self.spec.axes[0] == "channel":
                self.channel_labels = self.resolved_channel_labels(frame.shape[0])

        if self.writer is not None:
            self.writer.write(self, frame)

        self.notify()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def nbytes(self) -> int:
        """Memory held by every frame so far. A long run is what can
        exhaust a machine, so the data panel shows this as it grows. Kept as a
        running total because the panel asks on every append."""
        with self._lock:
            return self._nbytes

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def frames(self) -> list[np.ndarray]:
        """Every frame so far, oldest first."""
        with self._lock:
            return list(self._frames)

    def subscribe(self, callback: Callable[[Dataset], None]) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Dataset], None]) -> None:
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
        return f"<Dataset {self.output} frames={len(self)}>"
