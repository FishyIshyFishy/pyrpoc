"""A key -> class map with a registration decorator: how every plugin kind
is found. Each kind's registry lives in the structs file that defines the kind,
next to the base class it collects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")

# The decorated class itself, so ``@registry.register(...)`` keeps the class's
# own type downstream instead of a bare ``type``.
C = TypeVar("C", bound=type)


class Registry(Generic[T]):
    def __init__(self, name: str, base_class: type[T], *, stamp: str | None = "registry_key"):
        self.name = name
        self.base_class = base_class
        # The class attribute to record the key on, or None to leave the class alone.
        self.stamp = stamp
        self.entries: dict[str, type[T]] = {}

    def register(self, key: str) -> Callable[[C], C]:
        def decorator(cls: C) -> C:
            self.add(key, cls)
            return cls

        return decorator

    def add(self, key: str, cls: type) -> None:
        # The issubclass check is what narrows ``cls`` to ``type[T]``.
        if not issubclass(cls, self.base_class):
            raise TypeError(f"{cls.__name__} must inherit from {self.base_class.__name__}")
        if key in self.entries:
            raise KeyError(f"{key!r} is already registered in {self.name}")
        self.entries[key] = cls
        if self.stamp is not None:
            setattr(cls, self.stamp, key)

    def keys(self) -> list[str]:
        return sorted(self.entries)

    def get(self, key: str) -> type[T]:
        if key not in self.entries:
            raise KeyError(f"{key!r} is not registered in {self.name}")
        return self.entries[key]

    def key_for(self, cls: type) -> str:
        for key, registered in self.entries.items():
            if registered is cls:
                return key
        raise KeyError(f"{cls.__name__!r} is not registered in {self.name}")
