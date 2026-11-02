"""The open datasets, as everything outside the data library reads them."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from .data import Data
from .dataset import Dataset

D = TypeVar("D", bound=Data)


class Library(Protocol):
    """The open datasets, as panels and parameter editors see them."""

    def add(self, dataset: Dataset[D]) -> Dataset[D]: ...

    def by_id(self, dataset_id: str, kind: type[D]) -> Dataset[D] | None:
        """The open dataset with this id, if it holds ``kind``."""
        ...

    def matching(self, kind: type[D]) -> list[Dataset[D]]:
        """Datasets of this kind, newest first."""
        ...

    def subscribe(self, callback: Callable[[], None]) -> None: ...

    def unsubscribe(self, callback: Callable[[], None]) -> None: ...
