"""Recordings, and series of runs that continue one.

A recording is the datasets and optional saver that runs write into: one
library entry per output, one set of files. A run on its own gets a fresh
recording that closes when the run ends. Runs in a ``Series`` continue the
series' recording for as long as its ``RecordingKey`` holds; a run whose
parameters, devices or save target differ opens a new one, and the old one
closes. Closing a recording finalizes its datasets and files, once, after the
last run writing into it has ended.

Qt-free. The executor's lock guards every mutable field here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyrpoc.data_library.saving import RecordingSaver
from pyrpoc.structs.dataset import Dataset
from pyrpoc.structs.saving import SaveTarget

if TYPE_CHECKING:  # pragma: no cover
    from .executor import Run


@dataclass(frozen=True)
class RecordingKey:
    """What a recording's data depends on. A run whose key differs writes a
    new recording rather than mixing metadata into this one."""

    program_key: str
    parameters: dict[str, Any]
    devices: dict[str, Any]
    save: SaveTarget


class Recording:
    def __init__(
        self, key: RecordingKey, datasets: dict[str, Dataset], saver: RecordingSaver | None
    ):
        self.key = key
        self.datasets = datasets
        self.saver = saver
        # Runs writing into it right now; it closes only once this is zero.
        self.active = 0
        self.closed = False
        self.error: Exception | None = None
        # Where a failure to close is reported: the run that wrote last.
        self.last_run: Run | None = None

    def finalize(self) -> list[str]:
        """Close the datasets, then the saver, which describes them. Failures
        are collected rather than raised: a failed save is still a failure,
        and the other closers must still run."""
        closers: list[Callable[[], None]] = [
            lambda d=dataset: d.finalize(self.error) for dataset in self.datasets.values()
        ]
        saver = self.saver
        if saver is not None:
            closers.append(lambda: saver.finalize(self.error))
        errors: list[str] = []
        for close in closers:
            try:
                close()
            except Exception as exc:
                errors.append(str(exc))
        return errors


class Series:
    """Consecutive runs that continue one recording. The runner that opened it
    closes it; the recording closes once its last run has ended."""

    def __init__(self) -> None:
        self.recording: Recording | None = None
        self.closed = False
