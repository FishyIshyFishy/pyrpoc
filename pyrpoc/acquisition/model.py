"""What is set up to run: the selected program, its parameters, where it saves,
and whether anything is missing. The Acquisition panel shows it and changes it
through its commands; it holds no widgets."""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.data_library.model import LibraryModel
from pyrpoc.device_inventory.inventory import DeviceInventory
from pyrpoc.structs import params as P
from pyrpoc.structs.params import block_registry
from pyrpoc.structs.program import program_registry
from pyrpoc.structs.saving import SaveTarget

from . import claims
from .events import RunEvents
from .host import RunnerHost


class Acquisition(QObject):
    program_selected = pyqtSignal(str)
    save_changed = pyqtSignal()
    # Blocks changed outside the form, by a runner applying a pick.
    params_written = pyqtSignal()
    # Something worth remembering changed.
    edited = pyqtSignal()

    def __init__(self, inventory: DeviceInventory, library: LibraryModel, parent: QObject):
        super().__init__(parent)
        self.inventory = inventory
        self.library = library
        self.selected_program: str | None = None
        # One instance per block class, shared by every program declaring it,
        # so switching modality keeps the settings you made.
        self.blocks = P.BlockStore()
        # One save target for every program: where a run goes does not depend
        # on which program runs.
        self.save = SaveTarget()
        self.events = RunEvents(library.store, self)
        self.host = RunnerHost(self)
        # A run started with these settings, so they are worth keeping.
        self.events.run_started.connect(lambda _run: self.edited.emit())

    def select_program(self, key: str) -> None:
        self.selected_program = key
        self.params_for(key)
        self.host.attach(key, program_registry.get(key))
        self.program_selected.emit(key)
        self.edited.emit()

    def params_for(self, key: str) -> list[P.Group]:
        """The blocks one program declares, in declaration order, created at
        defaults on first request and shared from then on."""
        return [self.blocks.get(cls) for cls in program_registry.get(key).params]

    def blockers(self) -> list[str]:
        """What has to be supplied before the selected program can run. A
        missing name is here too, so the play button says why it is grey
        instead of throwing."""
        missing: list[str] = []
        if self.selected_program is not None:
            uses = list(program_registry.get(self.selected_program).uses)
            missing = [cls.display_name for cls in claims.missing(uses, self.inventory.devices)]
        if self.save.enabled and not self.save.filename:
            missing.append("a name to save under")
        return missing

    def set_save(
        self,
        *,
        name: str | None = None,
        directory: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Change only the fields given, so each launcher widget can write its
        own without reading the others back."""
        if name is not None:
            self.save.name = name
        if directory is not None:
            self.save.directory = directory
        if enabled is not None:
            self.save.enabled = enabled
        self.save_changed.emit()
        self.edited.emit()

    def mark_edited(self) -> None:
        """A parameter was edited in the form."""
        self.edited.emit()

    def params_state(self) -> dict[str, dict]:
        """The block store as a flat state dict, keyed by block class name."""
        return self.blocks.to_dict()

    def load_params_state(self, raw: dict[str, dict]) -> None:
        self.blocks.load_dict(raw, block_registry.entries)
