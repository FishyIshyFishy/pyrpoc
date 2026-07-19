"""Modality base and registry.

A modality declares a manifest, is configured with values + bound instruments +
enabled modifiers, and implements acquire_frame. The frame loop lives here so every
modality gets start/stop/limit handling for free.
"""

from __future__ import annotations

from pyrpoc_next.acquisition.modifiers.base import Modifier
from pyrpoc_next.acquisition.runner import RunContext
from pyrpoc_next.instruments.base import Instrument
from pyrpoc_next.structs.keys import InstrumentKey, ModalityKey
from pyrpoc_next.structs.manifest import ModalityManifest
from pyrpoc_next.structs.parcels import Parcel


class Modality:
    """Base for a way of acquiring data."""

    key: ModalityKey
    manifest: ModalityManifest

    def __init__(self):
        self.values: dict = {}
        self.instruments: dict[InstrumentKey, Instrument] = {}
        self.modifiers: list[Modifier] = []

    def configure(self, values: dict, instruments: dict[InstrumentKey, Instrument],
                  modifiers: list[Modifier]) -> None:
        """Bind coerced parameter values, the required instruments, and enabled modifiers."""
        self.values = values
        self.instruments = instruments
        self.modifiers = modifiers

    def run(self, context: RunContext) -> None:
        """Emit frames until stopped or the frame limit is reached."""
        index = 0
        while not context.should_stop():
            if context.frame_limit is not None and index >= context.frame_limit:
                break
            for parcel in self.acquire_frame(index):
                context.emit(parcel)
            index += 1

    def acquire_frame(self, index: int) -> list[Parcel]:
        """Acquire one frame and return the parcels it produced."""
        raise NotImplementedError


class ModalityRegistry:
    """Maps a ModalityKey to its Modality class."""

    def __init__(self):
        self.entries: dict[ModalityKey, type[Modality]] = {}

    def register(self, cls: type[Modality]) -> type[Modality]:
        self.entries[cls.key] = cls
        return cls

    def create(self, key: ModalityKey) -> Modality:
        return self.entries[key]()

    def manifest(self, key: ModalityKey) -> ModalityManifest:
        return self.entries[key].manifest

    def available(self) -> list[ModalityKey]:
        return list(self.entries)


modality_registry = ModalityRegistry()
