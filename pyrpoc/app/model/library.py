"""The data library as the screen sees it: what is open, and the commands on it.

The runtime ``DataLibrary`` holds the entries; this is where the Data Library
panel's actions land, so the panel draws and forwards and keeps no bookkeeping
of its own. It is also the ``Library`` every dataset panel and parameter editor
reads, so everything that reaches the library goes through one object.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from pyrpoc.structs.data import Data, Dataset

from ..runtime.library import DataLibrary, LibraryFull
from ..runtime.recording_format import RecordingError, load_recording, write_notes
from ..runtime.runs import Runs


class LibraryModel(QObject):
    auto_purge_changed = pyqtSignal(bool)
    # Why a recording could not be loaded. Reported, not raised: every caller
    # is a Qt slot.
    load_failed = pyqtSignal(str)

    def __init__(self, store: DataLibrary, runs: Runs, parent: QObject):
        super().__init__(parent)
        self.store = store
        self.runs = runs
        # Purging closes entries, which redraws panels, so it runs on the GUI
        # thread; coalesced so a burst of frames checks once. Runs never wait
        # on this thread, so it cannot stall acquisition.
        self._purge_timer = QTimer(self)
        self._purge_timer.setSingleShot(True)
        self._purge_timer.setInterval(0)
        self._purge_timer.timeout.connect(self.purge)
        runs.dataset_changed.connect(lambda _dataset: self.schedule_purge())
        store.subscribe(self.schedule_purge)

    def add(self, dataset: Dataset) -> Dataset:
        return self.store.add(dataset)

    def by_id(self, dataset_id: str) -> Dataset | None:
        return self.store.by_id(dataset_id)

    def matching(self, *specs: type[Data]) -> list[Dataset]:
        return self.store.matching(*specs)

    def all(self) -> list[Dataset]:
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
            datasets = load_recording(meta_path)
        except (LibraryFull, RecordingError) as exc:
            self.load_failed.emit(str(exc))
            return
        for dataset in datasets:
            self.store.add(dataset)

    def set_notes(self, dataset: Dataset, notes: str) -> str | None:
        """Set ``dataset``'s notes, and write them into its recording if that
        is on disk and finished; a recording still going writes them itself
        the next time it updates its metadata. Returns why the file could not
        be updated, or None; the notes are kept in memory either way."""
        dataset.notes = notes
        if dataset.meta_path is None or not dataset.finished:
            return None
        try:
            write_notes(dataset.meta_path, dataset.output, notes)
        except (RecordingError, OSError) as exc:
            return str(exc)
        return None

    def close(self, dataset: Dataset) -> None:
        """Drop ``dataset`` from memory. Files already saved stay on disk."""
        self.runs.stop_relaying(dataset)
        self.store.remove(dataset)

    def schedule_purge(self) -> None:
        if self.store.auto_purge and not self._purge_timer.isActive():
            self._purge_timer.start()

    def purge(self) -> None:
        """Close the oldest purgeable entries, one at a time, until under the
        limit. Stops early when only live or drawn entries are left."""
        while self.store.auto_purge and self.store.over_limit:
            candidates = self.store.purgeable()
            if not candidates:
                return
            self.close(candidates[0])
