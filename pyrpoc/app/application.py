"""The application: builds the device inventory, the data library and
acquisition, and connects them."""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.engine.acquisition.model import Acquisition
from pyrpoc.engine.data_library.model import LibraryModel
from pyrpoc.engine.data_library.store import LIBRARY_LIMIT_BYTES, LibraryStore
from pyrpoc.engine.device_inventory.inventory import DeviceInventory

# Importing the plugins registers every device, program and panel.
from pyrpoc.plugins import program_registry


class Application(QObject):
    # Anything worth autosaving.
    state_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.inventory = DeviceInventory(self)
        self.library = LibraryModel(LibraryStore(LIBRARY_LIMIT_BYTES), self)
        self.acquisition = Acquisition(self.inventory, self.library, self)
        # A restored session replaces this with the program it had selected.
        self.acquisition.select_program(program_registry.keys()[0])

        self.inventory.edited.connect(self.state_changed.emit)
        self.library.auto_purge_changed.connect(lambda _on: self.state_changed.emit())
        self.acquisition.edited.connect(self.state_changed.emit)
