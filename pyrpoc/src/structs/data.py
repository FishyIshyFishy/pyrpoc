"""The kinds of data the application passes around, and what holds them.

A ``Data`` subclass says what shape and dtype an array has and what its axes
mean -- ``Image2D`` is one. A program names each of its outputs in ``emits``
and says which kind it carries; a panel declares which kinds it can render; the
library filters on the kind. So a binding can be checked before a run starts
rather than inferred from a tag mid-flight.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
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
                f"{cls.name} expects {cls.ndim} dimensions {cls.axes}, "
                f"got shape {arr.shape}"
            )
        if any(size <= 0 for size in arr.shape):
            raise ValueError(f"{cls.name} received an empty axis: shape {arr.shape}")

    @classmethod
    def coerce(cls, array: np.ndarray) -> np.ndarray:
        cls.validate(array)
        return np.asarray(array, dtype=cls.dtype)


class Image2D(Data):
    """``(C, H, W)`` float32 — one image per channel."""

    name = "Image2D"
    ndim = 3
    axes = ("channel", "y", "x")


class Cube3D(Data):
    """``(H, W, B)`` float32 — one value per pixel per bin (FLIM histograms)."""

    name = "Cube3D"
    ndim = 3
    axes = ("y", "x", "bin")


class Samples4D(Data):
    """``(C, H, W, S)`` float32 — per-pixel raw samples, unaveraged.

    Split confocal's raw per-pixel samples. The design document files this as
    ``Image2D``; the array ``reshape_to_split_frame`` returns is four
    dimensional, so it gets its own contract rather than a false one.
    """

    name = "Samples4D"
    ndim = 4
    axes = ("channel", "y", "x", "sample")


class Spectrum1D(Data):
    """``(C, W)`` float32 — one spectrum per channel.

    Channel-first like ``Image2D`` rather than a bare ``(W,)``, for two
    reasons: ``axes[0] == "channel"`` is what makes ``Dataset.append`` fill in
    channel labels, and a spectrometer with more than one detector needs no new
    contract. A single-detector run is one channel, not a different shape.

    The spectral axis carries no units. What a bin means is the spectrometer's
    calibration, which is a device property rather than a shape contract -- so
    it belongs in dataset metadata when there is a real instrument to read it
    from.
    """

    name = "Spectrum1D"
    ndim = 2
    axes = ("channel", "wavelength")


class Mask2D(Data):
    """``(H, W)`` uint8 -- an authored region, 0 outside and non-zero inside.

    Its own contract rather than a one-channel ``Image2D``, and that is what
    makes the Modulation picker work: a mask is chosen with
    ``DataLibrary.matching(Mask2D)``, so a contract shared with acquired
    images would offer every run as a mask. The spec *is* the filter.

    uint8 rather than float32 because a mask is a decision per pixel, not a
    measurement. Nothing downstream minds -- the writers and
    ``normalize_channels`` cast for themselves.
    """

    name = "Mask2D"
    ndim = 2
    axes = ("y", "x")
    dtype = np.uint8


# --------------------------------------------------------------------------- #
# Datasets: where arrays of one kind live                                      #
# --------------------------------------------------------------------------- #


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    """One named output of one run: frames of one kind of ``Data``."""

    def __init__(
        self,
        *,
        output: str,
        spec: type[Data],
        provenance: Provenance,
        channel_labels: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        writer: Any | None = None,
    ):
        self.id = f"{output}-{uuid4().hex[:12]}"
        self.output = output
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
        parts = (self.started_time, self.name, self.output)
        return " · ".join(part for part in parts if part)

    def resolved_channel_labels(self, count: int) -> list[str]:
        if self.channel_labels and len(self.channel_labels) == count:
            return list(self.channel_labels)
        return [f"channel_{index}" for index in range(count)]

    # -- writing ----------------------------------------------------------- #

    def append(self, array: np.ndarray) -> None:
        """Validate, store, save, then notify.

        Runs on the worker thread, so subscriber callbacks do too. Nothing that
        touches Qt subscribes directly -- ``app/run_bridge.py`` is the only
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
        return f"<Dataset {self.output} frames={len(self)}>"
