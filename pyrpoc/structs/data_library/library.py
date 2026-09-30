"""The open datasets, as everything outside the data library reads them."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .data import Data
from .dataset import Dataset


class Library(Protocol):
    """The open datasets, as panels and parameter editors see them."""

    def add(self, dataset: Dataset) -> Dataset: ...

    def by_id(self, dataset_id: str) -> Dataset | None: ...

    def matching(self, *specs: type[Data]) -> list[Dataset]:
        """Datasets of these kinds, newest first."""
        ...

    def subscribe(self, callback: Callable[[], None]) -> None: ...

    def unsubscribe(self, callback: Callable[[], None]) -> None: ...
