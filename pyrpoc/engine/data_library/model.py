"""``LibraryModel``: the open recordings as the screen sees them, the commands on them, and
their outputs' changes relayed to the GUI thread."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from pyrpoc.core.recording import Recording

from .format import RecordingError, load_recording, recordings_in
from .notes import save_note
from .store import LibraryFull, LibraryStore


class LibraryModel(QObject):
    # Which recordings are open, or what they hold, changed. Emitted from the thread that changed
    # the store; Qt delivers it on the GUI thread, which is the only one panels may touch.
    changed = pyqtSignal()
    # A program output changed, delivered the same way.
    output_changed = pyqtSignal(object)
    auto_purge_changed = pyqtSignal(bool)
    # Why a recording could not be loaded. Reported, not raised: every caller
    # is a Qt slot.
    load_failed = pyqtSignal(str)

    def __init__(self, store: LibraryStore, parent: QObject):
        super().__init__(parent)
        self.store = store
        # Experiment folders opened whole, kept so one with no recordings still shows.
        self.folders: list[Path] = []
        # Purging closes entries, which redraws panels, so it runs on the GUI
        # thread; coalesced so a burst of changes checks once. Runs never wait
        # on this thread, so it cannot stall acquisition.
        self._purge_timer = QTimer(self)
        self._purge_timer.setSingleShot(True)
        self._purge_timer.setInterval(0)
        self._purge_timer.timeout.connect(self.purge)
        self.output_changed.connect(lambda _output: self.schedule_purge())
        self.changed.connect(self.schedule_purge)
        # Called at once, on the thread that changed the store, so a new output is followed
        # before its run adds to it. Subscribing to an output is safe from any thread.
        store.subscribe(self.relay_new_outputs)
        store.subscribe(self.changed.emit)

    def relay_new_outputs(self) -> None:
        """Follow every output of every open recording."""
        # Subscribing twice is a no-op, so this need not know which one was just added.
        for recording in self.store.recordings():
            for program_output in recording.outputs.values():
                program_output.subscribe(self.output_changed.emit)

    def add(self, recording: Recording) -> None:
        self.store.add(recording)

    def recordings(self) -> list[Recording]:
        return self.store.recordings()

    def subscribe(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` on the GUI thread after every change."""
        self.changed.connect(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        self.changed.disconnect(callback)

    @property
    def nbytes(self) -> int:
        return self.store.nbytes

    @property
    def limit_bytes(self) -> int:
        return self.store.limit_bytes

    @property
    def over_limit(self) -> bool:
        return self.store.over_limit

    @property
    def auto_purge(self) -> bool:
        return self.store.auto_purge

    def set_auto_purge(self, enabled: bool) -> None:
        if enabled == self.store.auto_purge:
            return
        self.store.auto_purge = enabled
        self.auto_purge_changed.emit(enabled)
        self.schedule_purge()

    def load(self, meta_path: Path) -> None:
        """Open a saved recording: every output becomes an entry. Refused while
        the library is full, as a new acquisition would be."""
        try:
            self.load_recording(meta_path)
        except (LibraryFull, RecordingError) as exc:
            self.load_failed.emit(str(exc))

    def load_experiment(self, folder: Path) -> None:
        """Open every recording in ``folder``, reporting the ones that could not be read
        together. Stops once the library is full."""
        if folder not in self.folders:
            self.folders.append(folder)
        problems: list[str] = []
        for meta_path in recordings_in(folder):
            try:
                self.load_recording(meta_path)
            except LibraryFull as exc:
                problems.append(str(exc))
                break
            except RecordingError as exc:
                problems.append(str(exc))
        # An empty folder adds no entry, yet it is now shown.
        self.store.notify()
        if problems:
            self.load_failed.emit("\n\n".join(problems))

    def load_recording(self, meta_path: Path) -> None:
        """Open the recording, unless it is already open."""
        if any(each.location == meta_path for each in self.store.recordings()):
            return
        self.store.check_room()
        self.store.add(load_recording(meta_path))

    def set_note(self, recording: Recording, note: str) -> str | None:
        """Set the recording's note and save it where the recording is saved; returns why the
        files could not be updated, or None."""
        # Kept in memory even when the files cannot be written.
        recording.note = note
        try:
            save_note(recording)
        except (RecordingError, OSError) as exc:
            return str(exc)
        return None

    def close(self, recording: Recording) -> None:
        """Drop ``recording`` from memory. Files already saved stay on disk."""
        for program_output in recording.outputs.values():
            program_output.unsubscribe(self.output_changed.emit)
        self.store.close(recording)

    def close_experiment(self, folder: Path) -> None:
        """Drop every recording saved in ``folder`` and stop showing it."""
        for recording in self.store.recordings():
            if recording.folder == folder:
                self.close(recording)
        if folder in self.folders:
            self.folders.remove(folder)
        self.store.notify()

    def schedule_purge(self) -> None:
        if self.store.auto_purge and not self._purge_timer.isActive():
            self._purge_timer.start()

    def purge(self) -> None:
        """Close the oldest purgeable entries, one at a time, until under the limit or only
        live entries are left."""
        while self.store.auto_purge and self.store.over_limit:
            candidates = self.store.purgeable()
            if not candidates:
                return
            self.close(candidates[0])
