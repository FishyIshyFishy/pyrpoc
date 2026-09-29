"""A key -> class map with a registration decorator, and the registries.

Implementations register themselves by importing their registry from here. The
panel registry lives in ``panel.py`` so registering a device never imports Qt.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from .data import Writer
from .device import Device
from .params import Group
from .program import Program

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


device_registry: Registry[Device] = Registry("DeviceRegistry", Device)

# Programs keep their key at the registration site, not on the class.
program_registry: Registry[Program] = Registry("ProgramRegistry", Program, stamp=None)

# Keyed by the name of the kind of ``Data`` saved; one writer may take several.
writer_registry: Registry[Writer] = Registry("WriterRegistry", Writer, stamp=None)

# Keyed by class name, a block's identity in the form, workspace and metadata.
block_registry: Registry[Group] = Registry("BlockRegistry", Group, stamp=None)

B = TypeVar("B", bound=type)


def block(cls: B) -> B:
    """Register a parameter block under its class name."""
    return block_registry.register(cls.__name__)(cls)
