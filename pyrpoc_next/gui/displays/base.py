"""Display base and registry.

A display is a Qt widget that also satisfies the core's DisplaySink protocol
(manifest + render). render() is called on the acquisition worker thread, so it just
emits a signal; the real drawing runs on the GUI thread in handle().
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget

from pyrpoc_next.structs.keys import DisplayKey
from pyrpoc_next.structs.manifest import DisplayManifest
from pyrpoc_next.structs.parcels import Parcel


class DisplayWidget(QWidget):
    """Base for a parcel-consuming display."""

    manifest: DisplayManifest
    parcel_received = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.parcel_received.connect(self.handle)

    def render(self, parcel: Parcel) -> None:
        """DisplaySink entry point (worker thread): hand off to the GUI thread."""
        self.parcel_received.emit(parcel)

    def handle(self, parcel: Parcel) -> None:
        """Draw a parcel on the GUI thread."""
        raise NotImplementedError


class DisplayRegistry:
    """Maps a DisplayKey to its display widget class."""

    def __init__(self):
        self.entries: dict[DisplayKey, type[DisplayWidget]] = {}

    def register(self, cls: type[DisplayWidget]) -> type[DisplayWidget]:
        self.entries[cls.manifest.key] = cls
        return cls

    def create(self, key: DisplayKey) -> DisplayWidget:
        return self.entries[key]()

    def available(self) -> list[DisplayKey]:
        return list(self.entries)


display_registry = DisplayRegistry()
