"""Instrument registry, keyed by InstrumentKey (no strings)."""

from __future__ import annotations

from pyrpoc_next.instruments.base import Instrument
from pyrpoc_next.structs.keys import InstrumentKey


class InstrumentRegistry:
    """Maps an InstrumentKey to its Instrument class."""

    def __init__(self):
        self.entries: dict[InstrumentKey, type[Instrument]] = {}

    def register(self, cls: type[Instrument]) -> type[Instrument]:
        """Class decorator: register by the class's declared key."""
        self.entries[cls.key] = cls
        return cls

    def create(self, key: InstrumentKey) -> Instrument:
        return self.entries[key]()

    def available(self) -> list[InstrumentKey]:
        return list(self.entries)


instrument_registry = InstrumentRegistry()
