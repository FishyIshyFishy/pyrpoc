"""The data library as the screen sees it: what is open, and the commands on it.

The runtime ``DataLibrary`` holds the entries; this is where the Data Library
panel's actions land, so the panel draws and forwards and keeps no bookkeeping
of its own. It is also the ``Library`` every dataset panel and parameter editor
reads, so everything that reaches the library goes through one object.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject

from pyrpoc.structs.data import Data, Dataset

from ..runtime.library import DataLibrary
from ..runtime.runs import Runs


class LibraryModel(QObject):
    def __init__(self, store: DataLibrary, runs: Runs, parent: QObject):
        super().__init__(parent)
        self.store = store
        self.runs = runs

    def add(self, dataset: Dataset) -> Dataset:
        return self.store.add(dataset)

    def by_id(self, dataset_id: str) -> Dataset | None:
        return self.store.by_id(dataset_id)

    def matching(self, *specs: type[Data]) -> list[Dataset]:
        return self.store.matching(*specs)

    def all(self) -> list[Dataset]:
        return self.store.all()

    def subscribe(self, callback: Callable[[], None]) -> None:
        self.store.subscribe(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        self.store.unsubscribe(callback)

    @property
    def nbytes(self) -> int:
        return self.store.nbytes

    def close(self, dataset: Dataset) -> None:
        """Drop ``dataset`` from memory. Files already saved stay on disk."""
        self.runs.stop_relaying(dataset)
        self.store.remove(dataset)
