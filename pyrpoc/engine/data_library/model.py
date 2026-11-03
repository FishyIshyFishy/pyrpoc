"""``LibraryModel``: the open program outputs as the screen sees them, the commands on
them, and their new values relayed to the GUI thread."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from pyrpoc.core.data import Data
from pyrpoc.core.program_output import ProgramOutput

from .format import RecordingError, load_recording, write_notes
from .store import LibraryFull, LibraryStore

D = TypeVar("D", bound=Data)


class LibraryModel(QObject):
    # A program output got a new value. Emitted from the thread that published; Qt
    # delivers it on the GUI thread, which is the only one panels may touch.
    output_changed = pyqtSignal(object)
    auto_purge_changed = pyqtSignal(bool)
    # Why a recording could not be loaded. Reported, not raised: every caller
    # is a Qt slot.
    load_failed = pyqtSignal(str)

    def __init__(self, store: LibraryStore, parent: QObject):
        super().__init__(parent)
        self.store = store
        # Purging closes entries, which redraws panels, so it runs on the GUI
        # thread; coalesced so a burst of publishes checks once. Runs never wait
        # on this thread, so it cannot stall acquisition.
        self._purge_timer = QTimer(self)
        self._purge_timer.setSingleShot(True)
        self._purge_timer.setInterval(0)
        self._purge_timer.timeout.connect(self.purge)
        self.output_changed.connect(lambda _output: self.schedule_purge())
        store.subscribe(self.relay_new_outputs)
        store.subscribe(self.schedule_purge)

    def relay_new_outputs(self) -> None:
        """Follow every open program output."""
        # Subscribing twice is a no-op, so this need not know which one was just added.
        for program_output in self.store.all():
            program_output.subscribe(self.output_changed.emit)

    def add(self, program_output: ProgramOutput[D]) -> ProgramOutput[D]:
        return self.store.add(program_output)

    def by_id(self, output_id: str, kind: type[D]) -> ProgramOutput[D] | None:
        return self.store.by_id(output_id, kind)

    def matching(self, kind: type[D]) -> list[ProgramOutput[D]]:
        return self.store.matching(kind)

    def all(self) -> list[ProgramOutput[Any]]:
        return self.store.all()

    def subscribe(self, callback: Callable[[], None]) -> None:
        self.store.subscribe(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        self.store.unsubscribe(callback)

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
            self.store.check_room()
            outputs = load_recording(meta_path)
        except (LibraryFull, RecordingError) as exc:
            self.load_failed.emit(str(exc))
            return
        for program_output in outputs:
            self.store.add(program_output)

    def set_notes(self, program_output: ProgramOutput[Any], notes: str) -> str | None:
        """Set ``program_output``'s notes, and write them into its recording if that is
        finished on disk; returns why the file could not be updated, or None."""
        # A recording still going writes the notes itself the next time it updates its metadata.
        # The notes are kept in memory either way.
        program_output.notes = notes
        if program_output.meta_path is None or not program_output.finished:
            return None
        try:
            write_notes(program_output.meta_path, program_output.name, notes)
        except (RecordingError, OSError) as exc:
            return str(exc)
        return None

    def close(self, program_output: ProgramOutput[Any]) -> None:
        """Drop ``program_output`` from memory. Files already saved stay on disk."""
        program_output.unsubscribe(self.output_changed.emit)
        self.store.remove(program_output)

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
