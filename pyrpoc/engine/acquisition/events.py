"""``RunEvents``: runs reported on the GUI thread."""

# Runs report on their worker threads; these signals carry that onto the GUI thread. They are the
# only subscriber to each program output a run opens, because a program adds to its outputs on
# the worker thread.

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.engine.acquisition.executor import RunCallbacks


class RunEvents(QObject):
    """What runs report. Each signal but ``start_refused`` carries its ``Run``."""

    run_started = pyqtSignal(object)
    run_status = pyqtSignal(object, str)
    run_failed = pyqtSignal(object, str)
    run_finished = pyqtSignal(object)
    # Refused before any run existed, by preflight: the message lists everything it needs.
    start_refused = pyqtSignal(str)

    def callbacks(self) -> RunCallbacks:
        """The executor's callbacks, each emitting its signal."""
        return RunCallbacks(
            on_status=self.run_status.emit,
            on_failed=self.run_failed.emit,
            on_finished=self.run_finished.emit,
        )
