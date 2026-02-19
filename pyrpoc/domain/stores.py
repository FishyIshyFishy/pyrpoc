from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class ObjectStore(Generic[T]):
    _items: list[T]

    def __init__(self, items: list[T]):
        self._items = items

    def add(self, item: T) -> T:
        self._items.append(item)
        return item

    def remove(self, item: T) -> None:
        if item in self._items:
            self._items.remove(item)

    def list(self) -> list[T]:
        return list(self._items)

    def contains(self, item: T) -> bool:
        return item in self._items

    def clear(self) -> list[T]:
        removed = list(self._items)
        self._items.clear()
        return removed

    def index_of(self, item: T) -> int:
        return self._items.index(item)
