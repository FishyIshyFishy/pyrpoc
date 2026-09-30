"""The datasets currently open, like a list of open documents. Not persisted.

Authored data is filed here too: a mask from the mask editor is a ``Mask2D``
entry, so a parameter can select it by kind without either side naming the
other. ``Dataset.origin`` says which kind of entry each one is.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from pyrpoc.structs.data import Data
from pyrpoc.structs.dataset import Dataset, Origin

# Everything open together, in bytes. Big enough for hours of typical imaging,
# small enough to leave a lab PC's RAM for the rest of the session.
LIBRARY_LIMIT_BYTES = 1 << 30


class LibraryFull(Exception):
    """Refused: more data would go past the limit, and auto-purge is off."""


def gib(nbytes: int) -> str:
    return f"{nbytes / (1 << 30):.2f} GiB"


class DataLibrary:
    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = limit_bytes
        # Close the oldest finished entries whenever the total is over the limit.
        self.auto_purge = False
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

    @property
    def over_limit(self) -> bool:
        return self.nbytes > self.limit_bytes

    def check_room(self) -> None:
        """Refuse new data while over the limit with nothing to free room."""
        if self.over_limit and not self.auto_purge:
            raise LibraryFull(
                f"The data library holds {gib(self.nbytes)}, over its "
                f"{gib(self.limit_bytes)} limit. Close entries in the Data Library "
                "panel, or turn on auto-purge there, to acquire or load more."
            )

    def purgeable(self) -> list[Dataset]:
        """What auto-purge may close, oldest first: finished entries, never one
        still being written and never one the user drew, which exists nowhere
        else."""
        with self._lock:
            return [
                dataset
                for dataset in self._datasets
                if dataset.finished and dataset.origin is not Origin.AUTHORED
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._datasets)
