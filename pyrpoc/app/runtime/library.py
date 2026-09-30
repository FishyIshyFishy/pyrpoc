"""The datasets currently open, like a list of open documents. Not persisted.

Authored data is filed here too: a mask from the mask editor is a ``Mask2D``
entry, so a parameter can select it by kind without either side naming the
other. ``Dataset.origin`` says which kind of entry each one is.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from pyrpoc.structs.data import Data, Dataset


class DataLibrary:
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
            self._datasets.remove(dataset)
        self.notify()

    def all(self) -> list[Dataset]:
        with self._lock:
            return list(self._datasets)

    def by_id(self, dataset_id: str) -> Dataset | None:
        with self._lock:
            return next((d for d in self._datasets if d.id == dataset_id), None)

    def matching(self, *specs: type[Data]) -> list[Dataset]:
        """Datasets of these kinds, newest first."""
        wanted = set(specs)
        with self._lock:
            return [d for d in reversed(self._datasets) if d.spec in wanted]

    def subscribe(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._subscribers.remove(callback)

    def notify(self) -> None:
        with self._lock:
            listeners = list(self._subscribers)
        for callback in listeners:
            callback()

    @property
    def nbytes(self) -> int:
        """Memory held by every entry together."""
        with self._lock:
            return sum(dataset.nbytes for dataset in self._datasets)

    def __len__(self) -> int:
        with self._lock:
            return len(self._datasets)
