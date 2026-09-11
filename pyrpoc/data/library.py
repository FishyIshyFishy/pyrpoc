"""The arrays currently open in the application, like a list of open documents.

Not persisted. The library starts empty each launch -- the same state the old
displays started in.

Authored data is filed here too, not only run outputs. A mask drawn in the mask
editor is added as a ``Mask2D`` entry, which is what lets the Modulation
parameter select one without the editor and that parameter knowing about each
other: the editor files a dataset, the parameter asks ``matching(Mask2D)``, and
neither names the other. The alternative was a path parameter and a save-then-
reload round trip through the filesystem.

Authored entries are distinguished two ways, both already here: by ``spec``,
which is what filters every picker, and by ``run_id == 0``, since no run
produced them.
"""

from __future__ import annotations

import threading
from typing import Callable

from pyrpoc.core.streams import Stream

from .dataset import Dataset


class DatasetLibrary:
    def __init__(self) -> None:
        self._datasets: list[Dataset] = []
        self._subscribers: list[Callable[[], None]] = []
        self._lock = threading.RLock()

    def add(self, dataset: Dataset) -> Dataset:
        with self._lock:
            self._datasets.append(dataset)
        self.notify()
        return dataset

    def remove(self, dataset: Dataset) -> None:
        with self._lock:
            if dataset not in self._datasets:
                return
            self._datasets.remove(dataset)
        self.notify()

    def clear(self) -> None:
        with self._lock:
            self._datasets.clear()
        self.notify()

    def all(self) -> list[Dataset]:
        with self._lock:
            return list(self._datasets)

    def by_id(self, dataset_id: str) -> Dataset | None:
        with self._lock:
            return next((d for d in self._datasets if d.id == dataset_id), None)

    def get(self, run_id: int, stream: str) -> Dataset | None:
        with self._lock:
            return next(
                (d for d in self._datasets if d.run_id == run_id and d.stream == stream), None
            )

    def matching(self, *specs: type[Stream]) -> list[Dataset]:
        """Datasets a view declaring these contracts could render, newest first."""
        wanted = set(specs)
        with self._lock:
            return [d for d in reversed(self._datasets) if d.spec in wanted]

    def subscribe(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def notify(self) -> None:
        with self._lock:
            listeners = list(self._subscribers)
        for callback in listeners:
            callback()

    def __len__(self) -> int:
        with self._lock:
            return len(self._datasets)
