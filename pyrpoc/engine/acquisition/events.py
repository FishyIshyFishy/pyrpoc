"""``RunEvents``: starts runs and reports them on the GUI thread."""

# Runs report on their worker threads; these signals carry that onto the GUI thread. They are the
# only subscriber to each program output a run opens, because a program adds to its outputs on
# the worker thread.

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.engine.acquisition.executor import Executor, Run, RunCallbacks
from pyrpoc.engine.acquisition.preflight import NotReady, Start
from pyrpoc.engine.data_library.store import LibraryStore


class RunEvents(QObject):
    """Starts runs and reports them. Each signal carries its ``Run``."""

    run_started = pyqtSignal(object)
    run_status = pyqtSignal(object, str)
    run_failed = pyqtSignal(object, str)
    run_finished = pyqtSignal(object)
    # Refused before any run existed, by preflight: the message lists everything it needs.
    start_refused = pyqtSignal(str)

    def __init__(self, library: LibraryStore, parent: QObject):
        super().__init__(parent)
        self.executor = Executor(
            library,
            RunCallbacks(
                on_status=self.run_status.emit,
                on_failed=self.run_failed.emit,
                on_finished=self.run_finished.emit,
            ),
        )

    def start(
        self, start: Start, note: str, repeat: bool, claim: Callable[[Run], None]
    ) -> Run | None:
        """Start the program and return its run, or report through ``start_refused`` why it
        cannot start."""
        # Reported rather than raised, since every caller is a Qt slot. ``claim`` gets the run
        # before ``run_started`` announces it, so whoever started it has recorded it by then.
        try:
            run = self.executor.start(start, note=note, repeat=repeat)
        except NotReady as exc:
            self.start_refused.emit(str(exc))
            return None
        claim(run)
        self.run_started.emit(run)
        return run
