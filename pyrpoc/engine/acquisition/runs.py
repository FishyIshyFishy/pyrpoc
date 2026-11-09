"""``Runs``: starting programs, where runs save, and which devices live runs hold."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.core.program import program_registry
from pyrpoc.engine.acquisition.save_target import SaveTarget
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
    # What a program needs may have arrived or gone: a device, a name to save under, an earlier
    # output, or a program input value its checks judge.
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
        self.earlier = EarlierChoices(library.store)
        self.events = RunEvents(library.store, self)
        self.live_runs: set[Run] = set()
        self.events.run_started.connect(self.on_run_started)
        self.events.run_finished.connect(self.on_run_finished)
        inventory.changed.connect(self.readiness_changed.emit)
        inventory.edited.connect(self.post_device_edits)
        self.save_changed.connect(self.readiness_changed.emit)
        self.earlier_changed.connect(self.readiness_changed.emit)
        inputs.subscribe(lambda _key: self.readiness_changed.emit())

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
            self.earlier.load,
            repeat,
            claim,
        )
        # Cleared so one note is never copied onto runs it was not written for.
        if run is not None and self.note:
            self.set_note("")
        # Saved under the next free name when the one asked for was taken, and shown so.
        if run is not None and run.saver is not None and run.recording.name != self.save.name:
            self.save.name = run.recording.name
            self.save_changed.emit()

    def blockers(self, program_key: str) -> list[str]:
        """What has to be supplied before ``program_key`` can run."""
        program = program_registry.get(program_key)
        start = readiness.StartingRun(
            self.save.folder,
            self.inputs.values_for(program.inputs),
            self.chosen_earlier(program_key),
            self.earlier.load,
        )
        return readiness.missing(program, self.inventory.devices, self.save, start)

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

    def pick_earlier(self, name: str, choice: EarlierOutput) -> None:
        """Read ``choice`` wherever ``name`` is read."""
        self.earlier.pick(name, choice)
        self.earlier_changed.emit()

    def set_save(
        self,
        *,
        name: str | None = None,
        directory: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Change only the fields given. Refused while a run is going, since a recording cannot
        move to another folder or name partway through."""
        if self.live_runs:
            raise RuntimeError("the save target cannot change while a run is going")
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
        # Here, on the thread program inputs are set on, so no edit lands mid-close.
        run.inbox.close()
        self.live_runs.discard(run)
        self.publish_in_use()
        # Only a saved run can change what later runs read.
        if run.saver is not None and any_reads():
            self.refresh_earlier()

    def post_device_edits(self) -> None:
        """Post an edit to the devices to every run going, which takes it between passes."""
        for run in self.live_runs:
            run.inbox.devices_edited()

    def publish_in_use(self) -> None:
        """Tell the inventory every device a live run holds, backing devices included."""
        self.inventory.set_in_use(
            {device for run in self.live_runs for device in run.devices.values()}
        )
