"""A key -> class map with a registration decorator.

Devices, views and programs each need one, and the key belongs at the
registration site rather than on the class -- which is what lets Program keep to
the four attributes section 12 allows it.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str, base_class: type, *, stamp: bool = True):
        self.name = name
        self.base_class = base_class
        #: Whether to record the key on the class. Off for programs, so a
        #: Program subclass keeps to the four attributes section 12 allows.
        self.stamp = stamp
        self.entries: dict[str, type[T]] = {}

    def register(self, key: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            if not issubclass(cls, self.base_class):
                raise TypeError(f"{cls.__name__} must inherit from {self.base_class.__name__}")
            if key in self.entries:
                raise KeyError(f"{key!r} is already registered in {self.name}")
            self.entries[key] = cls  # type: ignore[assignment]
            if self.stamp:
                cls.registry_key = key  # type: ignore[attr-defined]
            return cls

        return decorator

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
        raise KeyError(f"{getattr(cls, '__name__', cls)!r} is not registered in {self.name}")

    def create(self, key: str, **kwargs: Any) -> T:
        return self.get(key)(**kwargs)  # type: ignore[call-arg]
