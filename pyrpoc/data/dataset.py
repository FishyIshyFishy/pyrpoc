"""Where acquired arrays live.

A dataset is a run output: arrays appended as they arrive, a shape contract, the
parameters that produced it, and optionally a writer putting it on disk. Views
render datasets and never own arrays -- which is the fix for ``self._data_chw``
in the old displays, where the array *was* the data, closing a display destroyed
it, and two displays over one run held two drifting copies.

A run that acquires repeatedly is one dataset that grows. Appending is the whole
event: nothing outside a program is told which array this was or how many are
expected, because only the program that set the count knows.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from pyrpoc.core.streams import Stream


@dataclass(frozen=True)
class Provenance:
    """What produced this data. Written into the saved metadata."""

    program_key: str
    parameters: dict[str, Any] = field(default_factory=dict)
    devices: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    run_id: int = 0
    #: What the run was called -- the save filename, set whether or not the
    #: run was saved. An unnamed run falls back to its program.
    name: str = ""


class Dataset:
    """One named output stream of one run."""

    def __init__(
        self,
        *,
        stream: str,
        spec: type[Stream],
        provenance: Provenance,
        channel_labels: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        writer: Any | None = None,
    ):
        self.id = f"{stream}-{uuid4().hex[:12]}"
        self.stream = stream
        self.spec = spec
        self.provenance = provenance
        self.channel_labels: list[str] = list(channel_labels or [])
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.writer = writer

        self._frames: list[np.ndarray] = []
        self._nbytes = 0
        self._subscribers: list[Callable[["Dataset"], None]] = []
        self._lock = threading.RLock()

    # -- identity ---------------------------------------------------------- #

    @property
    def run_id(self) -> int:
        return self.provenance.run_id

    @property
    def name(self) -> str:
        return self.provenance.name or self.provenance.program_key

    @property
    def started_time(self) -> str:
        """Local time of day the run started, or "" if that is not known.

        No date. A session lasts a day, so a date column would repeat the same
        ten characters on every row and push the useful part off the edge.
        """
        raw = self.provenance.started_at
        if not raw:
            return ""
        try:
            moment = datetime.fromisoformat(raw)
        except ValueError:
            return ""
        if moment.tzinfo is not None:
            moment = moment.astimezone()
        return moment.strftime("%H:%M:%S")

    @property
    def label(self) -> str:
        """One line naming this dataset, for pickers that have no columns.

        The run id used to stand in for identity here. The time is what a user
        actually recognises a run by, and the name is what they chose.
        """
        parts = (self.started_time, self.name, self.stream)
        return " · ".join(part for part in parts if part)

    def resolved_channel_labels(self, count: int) -> list[str]:
        if self.channel_labels and len(self.channel_labels) == count:
            return list(self.channel_labels)
        return [f"channel_{index}" for index in range(count)]

    # -- writing ----------------------------------------------------------- #

    def append(self, array: np.ndarray) -> None:
        """Validate, store, save, then notify.

        Runs on the worker thread, so subscriber callbacks do too. Nothing that
        touches Qt subscribes directly -- ``shell/run_bridge.py`` is the only
        subscriber and it re-emits on a signal, which Qt queues to the GUI
        thread.
        """
        frame = self.spec.coerce(array)
        with self._lock:
            self._frames.append(frame)
            self._nbytes += frame.nbytes
            if not self.channel_labels and self.spec.axes and self.spec.axes[0] == "channel":
                self.channel_labels = self.resolved_channel_labels(frame.shape[0])

        if self.writer is not None:
            self.writer.write(self, frame)

        self.notify()

    # -- reading ----------------------------------------------------------- #

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def nbytes(self) -> int:
        """How much memory everything appended so far is holding.

        A dataset keeps every array it was given, so a long continuous run is
        the one thing here that can exhaust a machine. The data panel reports
        this so that is visible while it happens rather than afterwards.

        Accumulated in ``append`` rather than summed on demand: the panel asks
        again on every append, and summing a list that grows by one each time
        would make displaying the number quadratic in the length of the run.
        """
        with self._lock:
            return self._nbytes

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def stack(self) -> np.ndarray | None:
        """Every frame as one array with a leading frame axis."""
        with self._lock:
            if not self._frames:
                return None
            return np.stack(self._frames, axis=0)

    # -- change notification ------------------------------------------------ #

    def subscribe(self, callback: Callable[["Dataset"], None]) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[["Dataset"], None]) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def notify(self) -> None:
        with self._lock:
            listeners = list(self._subscribers)
        for callback in listeners:
            callback(self)

    # -- lifecycle ---------------------------------------------------------- #

    def finalize(self, error: Exception | None) -> None:
        if self.writer is not None:
            self.writer.finalize(self, error)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Dataset {self.stream} frames={len(self)}>"
