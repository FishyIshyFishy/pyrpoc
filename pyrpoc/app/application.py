"""The application: builds the device inventory, the data library, the program
inputs and the runs, and connects them."""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.engine.acquisition.program_inputs import ProgramInputsStore
from pyrpoc.engine.acquisition.runs import Runs
from pyrpoc.engine.data_library.model import LibraryModel
from pyrpoc.engine.data_library.store import LIBRARY_LIMIT_BYTES, LibraryStore
from pyrpoc.engine.device_inventory.inventory import DeviceInventory


class Application(QObject):
    """The app's working state: its three collections and the runs."""

    # Working state worth autosaving changed.
    state_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.inventory = DeviceInventory(self)
        self.library = LibraryModel(LibraryStore(LIBRARY_LIMIT_BYTES), self)
        self.inputs = ProgramInputsStore()
        self.runs = Runs(self.inventory, self.library, self.inputs, self)

        self.inventory.edited.connect(self.state_changed.emit)
        self.library.auto_purge_changed.connect(lambda _on: self.state_changed.emit())
        self.inputs.subscribe(lambda _key: self.state_changed.emit())
        self.runs.edited.connect(self.state_changed.emit)
