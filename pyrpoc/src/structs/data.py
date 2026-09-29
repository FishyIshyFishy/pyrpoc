"""The kinds of data the application passes around, and what holds them.

A ``Data`` subclass fixes an array's shape, dtype and axis meanings. Programs
name the kind each output carries and panels name the kinds they render, so a
binding is checked before a run starts rather than inferred mid-flight.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

import numpy as np


class Data:
    """Base kind. Subclasses fix ``ndim``, ``axes`` and a human name."""

    name: str = "data"
    ndim: int = 0
    axes: tuple[str, ...] = ()
    dtype = np.float32

    @classmethod
    def validate(cls, array: np.ndarray) -> None:
        arr = np.asarray(array)
        if arr.ndim != cls.ndim:
            raise ValueError(
                f"{cls.name} expects {cls.ndim} dimensions {cls.axes}, got shape {arr.shape}"
            )
        if any(size <= 0 for size in arr.shape):
            raise ValueError(f"{cls.name} received an empty axis: shape {arr.shape}")

    @classmethod
    def coerce(cls, array: np.ndarray) -> np.ndarray:
        cls.validate(array)
        return np.asarray(array, dtype=cls.dtype)


class Image2D(Data):
    """``(C, H, W)`` float32: one image per channel."""

    name = "Image2D"
    ndim = 3
    axes = ("channel", "y", "x")


class Cube3D(Data):
    """``(H, W, B)`` float32: one value per pixel per bin (FLIM histograms)."""

    name = "Cube3D"
    ndim = 3
    axes = ("y", "x", "bin")


class Samples4D(Data):
    """``(C, H, W, S)`` float32: per-pixel raw samples, unaveraged."""

    name = "Samples4D"
    ndim = 4
    axes = ("channel", "y", "x", "sample")


class Spectrum1D(Data):
    """``(C, W)`` float32: one spectrum per channel.

    Channel-first like ``Image2D`` so ``Dataset.append`` fills in channel
    labels and a multi-detector spectrometer needs no new kind. Bins carry no
    units: that is the spectrometer's calibration, not a shape contract.
    """

    name = "Spectrum1D"
    ndim = 2
    axes = ("channel", "wavelength")


class Mask2D(Data):
    """``(H, W)`` uint8: an authored region, 0 outside and non-zero inside.

    Its own kind rather than a one-channel ``Image2D``, because pickers filter
    on the kind: sharing one with acquired images would offer every run as a
    mask. uint8 because a mask is a decision per pixel, not a measurement.
    """

    name = "Mask2D"
    ndim = 2
    axes = ("y", "x")
    dtype = np.uint8


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


class Library(Protocol):
    """The open datasets, as panels and parameter editors see them."""

    def add(self, dataset: Dataset) -> Dataset: ...

    def by_id(self, dataset_id: str) -> Dataset | None: ...

    def matching(self, *specs: type[Data]) -> list[Dataset]:
        """Datasets of these kinds, newest first."""
        ...

    def subscribe(self, callback: Callable[[], None]) -> None: ...

    def unsubscribe(self, callback: Callable[[], None]) -> None: ...


class DatasetWriter(Protocol):
    """Where a dataset's frames go on disk as they arrive."""

    def write(self, dataset: Dataset, array: np.ndarray) -> None: ...

    def finalize(self, dataset: Dataset, error: Exception | None) -> None: ...


class Dataset:
    """One named output of one run: frames of one kind of ``Data``."""

    def __init__(
        self,
        *,
        output: str,
        spec: type[Data],
        provenance: Provenance,
        writer: DatasetWriter | None = None,
    ):
        self.id = f"{output}-{uuid4().hex[:12]}"
        self.output = output
        self.spec = spec
        self.provenance = provenance
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
        subscribers must not touch Qt; ``app/run_bridge.py`` re-emits for them."""
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
        """Memory held by every frame so far. A long continuous run is what can
        exhaust a machine, so the data panel shows this as it grows. Kept as a
        running total because the panel asks on every append."""
        with self._lock:
            return self._nbytes

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

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
        if self.writer is not None:
            self.writer.finalize(self, error)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Dataset {self.output} frames={len(self)}>"
