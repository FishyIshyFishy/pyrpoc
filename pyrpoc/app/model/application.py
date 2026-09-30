"""The application: builds the three parts and connects them.

    inventory     the devices this workbench has        (device_inventory/)
    library       the open data                         (data_library/)
    acquisition   the program set up to run, and runs   (acquisition/)

Importing ``pyrpoc.plugins`` here is what registers every device, program and
data panel; nothing else lists them. It holds no widgets.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.acquisition.model import Acquisition
from pyrpoc.data_library.model import LibraryModel
from pyrpoc.data_library.store import LIBRARY_LIMIT_BYTES, LibraryStore
from pyrpoc.device_inventory.inventory import DeviceInventory
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
