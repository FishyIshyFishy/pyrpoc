"""``Runs``: starting programs, where runs save, and which devices live runs hold."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.core.program import program_registry
from pyrpoc.core.saving import SaveTarget
from pyrpoc.engine.data_library.model import LibraryModel
from pyrpoc.engine.device_inventory.inventory import DeviceInventory

from . import claims
from .events import RunEvents
from .executor import Run
from .program_inputs import ProgramInputsStore
from .recording import Series


class Runs(QObject):
    """Starts runs of any program with the current program inputs, and keeps the
    one save target every run shares."""

    save_changed = pyqtSignal()
    note_changed = pyqtSignal()
    # What a program needs may have arrived or gone: a device, or a name to save under.
    readiness_changed = pyqtSignal()
    # Something worth remembering changed: the save target, or a run started.
    edited = pyqtSignal()

    def __init__(
        self,
        inventory: DeviceInventory,
        library: LibraryModel,
        inputs: ProgramInputsStore,
        parent: QObject,
    ):
        super().__init__(parent)
        self.inventory = inventory
        self.inputs = inputs
        self.save = SaveTarget()
        # The note for the next run. Run metadata, not a program input: no program reads it.
        self.note = ""
        self.events = RunEvents(library.store, self)
        self.live_runs: set[Run] = set()
        self.events.run_started.connect(self.on_run_started)
        self.events.run_finished.connect(self.on_run_finished)
        inventory.changed.connect(self.readiness_changed.emit)
        self.save_changed.connect(self.readiness_changed.emit)

    def start(self, program_key: str, series: Series | None, claim: Callable[[Run], None]) -> None:
        """Start ``program_key`` with what the workbench holds right now, or report
        why not through ``events.start_refused``. A run that starts takes the note."""
        run = self.events.start(
            program_registry.get(program_key)(),
            program_key,
            self.inputs,
            self.inventory.devices,
            self.save,
            self.note,
            series,
            claim,
        )
        # Cleared so one note is never copied onto runs it was not written for.
        if run is not None and self.note:
            self.set_note("")

    def blockers(self, program_key: str) -> list[str]:
        """What has to be supplied before ``program_key`` can run."""
        required = list(program_registry.get(program_key).devices)
        missing = [cls.display_name for cls in claims.missing(required, self.inventory.devices)]
        if self.save.enabled and self.save.folder is None:
            missing.append("an Experiment folder to save into")
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
        """Change only the fields given."""
        if name is not None:
            self.save.name = name
        if directory is not None:
            self.save.directory = directory
        if enabled is not None:
            self.save.enabled = enabled
        self.save_changed.emit()
        self.edited.emit()

    def set_note(self, text: str) -> None:
        self.note = text
        self.note_changed.emit()

    def on_run_started(self, run: Run) -> None:
        self.live_runs.add(run)
        self.publish_in_use()
        # A run started with these settings, so they are worth keeping.
        self.edited.emit()

    def on_run_finished(self, run: Run) -> None:
        self.live_runs.discard(run)
        self.publish_in_use()

    def publish_in_use(self) -> None:
        """Tell the inventory every device a live run holds, backing devices included."""
        self.inventory.set_in_use(
            {device for run in self.live_runs for device in run.devices.values()}
        )
