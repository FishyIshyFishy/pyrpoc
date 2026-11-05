"""``Runs``: starting programs, where runs save, and which devices live runs hold."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.core.program import program_registry
from pyrpoc.core.saving import SaveTarget
from pyrpoc.engine.data_library.grouping import program_display_name
from pyrpoc.engine.data_library.model import LibraryModel
from pyrpoc.engine.device_inventory.inventory import DeviceInventory

from . import claims
from .earlier import EarlierChoices, read_program_keys
from .events import RunEvents
from .executor import Run
from .program_inputs import ProgramInputsStore
from .recording import Series


class Runs(QObject):
    """Starts runs of any program with the current program inputs, and keeps the
    one save target every run shares."""

    save_changed = pyqtSignal()
    note_changed = pyqtSignal()
    # The recordings programs can read changed, or which one is picked.
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

    def start(self, program_key: str, series: Series | None, claim: Callable[[Run], None]) -> None:
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
            missing.append("a folder to save into")
        if self.save.enabled and not self.save.filename:
            missing.append("a name to save under")
        for each in program_registry.get(program_key).reads:
            if self.earlier.chosen(each.program_key) is None:
                missing.append(self.missing_earlier(each.program_key))
        return missing

    def missing_earlier(self, program_key: str) -> str:
        """How a blocker names an earlier run with nothing to read."""
        name = program_display_name(program_key)
        if self.save.folder is None:
            return f"a folder with {name} saved in it"
        return f"{name} saved in this folder"

    def chosen_earlier(self, program_key: str) -> dict[str, Path]:
        """The recording each earlier run ``program_key`` reads would be read from now."""
        chosen: dict[str, Path] = {}
        for each in program_registry.get(program_key).reads:
            path = self.earlier.chosen(each.program_key)
            if path is not None:
                chosen[each.program_key] = path
        return chosen

    def refresh_earlier(self) -> None:
        """Look again at the recordings in the save folder that programs read."""
        self.earlier.refresh(self.save.folder)
        self.earlier_changed.emit()

    def pick_earlier(self, program_key: str, meta_path: Path | None) -> None:
        """Read ``meta_path`` wherever ``program_key`` is read, or the latest when None."""
        self.earlier.pick(program_key, meta_path)
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
        # Only a saved run of a program others read can change what they read.
        saved = run.recording.saver is not None
        if saved and run.provenance.program_key in read_program_keys():
            self.refresh_earlier()

    def publish_in_use(self) -> None:
        """Tell the inventory every device a live run holds, backing devices included."""
        self.inventory.set_in_use(
            {device for run in self.live_runs for device in run.devices.values()}
        )
