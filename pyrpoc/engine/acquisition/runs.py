"""``Runs``: starting programs, where runs save, and which devices live runs hold."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QFileSystemWatcher, QObject, QTimer, pyqtSignal

from pyrpoc.core.program import program_registry
from pyrpoc.engine.acquisition.save_target import SaveTarget
from pyrpoc.engine.data_library.model import LibraryModel
from pyrpoc.engine.device_inventory.inventory import DeviceInventory

from .earlier import fill_earlier
from .events import RunEvents
from .executor import Run
from .preflight import Start
from .program_inputs import ProgramInputsStore


class Runs(QObject):
    """Starts runs of any program with the current program inputs, and keeps the
    one save target every run shares."""

    save_changed = pyqtSignal()
    note_changed = pyqtSignal()
    # Something preflight looks at changed: the devices or the runs holding them, where runs
    # save or what that folder holds, a program input, or the library's room.
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
        self.library = library
        # Why, by key, each earlier output input's newest in the save folder cannot be read.
        self.unreadable: dict[str, str] = {}
        self.events = RunEvents(library.store, self)
        self.live_runs: set[Run] = set()
        # Files land in the save folder in bursts as a run saves, so one look follows a burst.
        self.folder_watcher = QFileSystemWatcher(self)
        self.folder_settled = QTimer(self)
        self.folder_settled.setSingleShot(True)
        self.folder_settled.setInterval(300)
        self.folder_watcher.directoryChanged.connect(lambda _path: self.folder_settled.start())
        self.folder_settled.timeout.connect(self.refresh_earlier)
        self.events.run_started.connect(self.on_run_started)
        self.events.run_finished.connect(self.on_run_finished)
        inventory.changed.connect(self.readiness_changed.emit)
        inventory.edited.connect(self.post_device_edits)
        self.save_changed.connect(self.readiness_changed.emit)
        inputs.subscribe(lambda _key: self.readiness_changed.emit())
        # Called on whichever thread changed the library; the signal carries it to this one.
        library.subscribe(self.readiness_changed.emit)
        library.auto_purge_changed.connect(lambda _on: self.readiness_changed.emit())

    def start(self, program_key: str, repeat: bool, claim: Callable[[Run], None]) -> None:
        """Start ``program_key`` with what the workbench holds right now, or report
        why not through ``events.start_refused``. A run that starts takes the note."""
        run = self.events.start(self.start_of(program_key), self.note, repeat, claim)
        # Cleared so one note is never copied onto runs it was not written for.
        if run is not None and self.note:
            self.set_note("")
        # Saved under the next free name when the one asked for was taken, and shown so.
        if run is not None and run.saver is not None and run.recording.name != self.save.name:
            self.save.name = run.recording.name
            self.save_changed.emit()

    def blockers(self, program_key: str) -> list[str]:
        """Everything ``program_key`` still needs before it can run."""
        return self.events.executor.preflight(self.start_of(program_key))

    def start_of(self, program_key: str) -> Start:
        """A start of ``program_key`` with what the workbench holds right now."""
        program = program_registry.get(program_key)
        return Start(
            program_key=program_key,
            program=program(),
            inputs=self.inputs,
            inventory=self.inventory.devices,
            save=self.save,
            unreadable=self.unreadable,
        )

    def refresh_earlier(self) -> None:
        """Look again at the save folder: what it holds, and the earlier outputs in it."""
        self.watch_folder()
        self.unreadable = fill_earlier(self.save.folder, self.inputs, self.library.store)
        self.readiness_changed.emit()

    def watch_folder(self) -> None:
        """Follow the save folder, once it exists, so files moved in or out outside the app are
        judged again."""
        watched = self.folder_watcher.directories()
        folder = self.save.folder
        wanted = [str(folder)] if folder is not None and folder.is_dir() else []
        if watched != wanted:
            if watched:
                self.folder_watcher.removePaths(watched)
            if wanted:
                self.folder_watcher.addPaths(wanted)

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
        if directory is not None:
            self.save.directory = directory
        if enabled is not None:
            self.save.enabled = enabled
        # Also when the folder is the same: a restored session has just cleared every input.
        if directory is not None:
            self.refresh_earlier()
        self.save_changed.emit()
        self.edited.emit()

    def set_note(self, text: str) -> None:
        self.note = text
        self.note_changed.emit()

    def on_run_started(self, run: Run) -> None:
        self.live_runs.add(run)
        self.publish_in_use()
        self.readiness_changed.emit()
        # A run started with these settings, so they are worth keeping.
        self.edited.emit()

    def on_run_finished(self, run: Run) -> None:
        # Here, on the thread program inputs are set on, so no edit lands mid-close.
        run.inbox.close()
        self.live_runs.discard(run)
        self.publish_in_use()
        # A saved run changes what its folder holds, and may have just made the folder.
        if run.saver is not None:
            self.refresh_earlier()
        else:
            self.readiness_changed.emit()

    def post_device_edits(self) -> None:
        """Post an edit to the devices to every run going, which takes it between passes."""
        for run in self.live_runs:
            run.inbox.devices_edited()

    def publish_in_use(self) -> None:
        """Tell the inventory every device a live run holds, backing devices included."""
        self.inventory.set_in_use(
            {device for run in self.live_runs for device in run.devices.values()}
        )
