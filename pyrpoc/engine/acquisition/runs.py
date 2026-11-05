"""``Runs``: starting programs, where runs save, and which devices live runs hold."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.core.program import program_registry
from pyrpoc.core.saving import SaveTarget
from pyrpoc.engine.data_library.earlier import EarlierOutput
from pyrpoc.engine.data_library.model import LibraryModel
from pyrpoc.engine.device_inventory.inventory import DeviceInventory

from . import readiness
from .earlier import EarlierChoices, any_reads
from .events import RunEvents
from .executor import Run
from .program_inputs import ProgramInputsStore


class Runs(QObject):
    """Starts runs of any program with the current program inputs, and keeps the
    one save target every run shares."""

    save_changed = pyqtSignal()
    note_changed = pyqtSignal()
    # The earlier outputs programs can read changed, or which one is picked.
    earlier_changed = pyqtSignal()
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
        self.earlier = EarlierChoices()
        self.events = RunEvents(library.store, self)
        self.live_runs: set[Run] = set()
        self.events.run_started.connect(self.on_run_started)
        self.events.run_finished.connect(self.on_run_finished)
        inventory.changed.connect(self.readiness_changed.emit)
        self.save_changed.connect(self.readiness_changed.emit)
        self.earlier_changed.connect(self.readiness_changed.emit)

    def start(self, program_key: str, repeat: bool, claim: Callable[[Run], None]) -> None:
        """Start ``program_key`` with what the workbench holds right now, or report
        why not through ``events.start_refused``. A run that starts takes the note."""
        program = program_registry.get(program_key)()
        run = self.events.start(
            program,
            program_key,
            self.inputs,
            self.inventory.devices,
            self.save,
            self.note,
            self.chosen_earlier(program_key),
            repeat,
            claim,
        )
        # Cleared so one note is never copied onto runs it was not written for.
        if run is not None and self.note:
            self.set_note("")

    def blockers(self, program_key: str) -> list[str]:
        """What has to be supplied before ``program_key`` can run."""
        return readiness.missing(
            program_registry.get(program_key),
            self.inventory.devices,
            self.save,
            self.chosen_earlier(program_key),
        )

    def chosen_earlier(self, program_key: str) -> dict[str, EarlierOutput]:
        """The earlier output each name ``program_key`` reads would be read from now."""
        chosen: dict[str, EarlierOutput] = {}
        for name, output_type in program_registry.get(program_key).reads.items():
            choice = self.earlier.chosen(name, output_type)
            if choice is not None:
                chosen[name] = choice
        return chosen

    def refresh_earlier(self) -> None:
        """Look again at the earlier outputs in the save folder."""
        self.earlier.refresh(self.save.folder)
        self.earlier_changed.emit()

    def pick_earlier(self, name: str, choice: EarlierOutput | None) -> None:
        """Read ``choice`` wherever ``name`` is read, or the latest when None."""
        self.earlier.pick(name, choice)
        self.earlier_changed.emit()

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
        moved = directory is not None and directory != self.save.directory
        if directory is not None:
            self.save.directory = directory
        if enabled is not None:
            self.save.enabled = enabled
        if moved:
            self.refresh_earlier()
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
        # Here, on the thread program inputs are set on, so no change lands mid-close.
        run.live.close()
        self.live_runs.discard(run)
        self.publish_in_use()
        # Only a saved run can change what later runs read.
        if run.saver is not None and any_reads():
            self.refresh_earlier()

    def publish_in_use(self) -> None:
        """Tell the inventory every device a live run holds, backing devices included."""
        self.inventory.set_in_use(
            {device for run in self.live_runs for device in run.devices.values()}
        )
