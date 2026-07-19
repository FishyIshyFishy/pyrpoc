"""Marshals the core controller's worker-thread callbacks onto the GUI thread.

The controller calls on_stopped/on_error from the acquisition worker thread; these
signals cross back to the GUI thread so widgets update safely.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


class RunBridge(QObject):
    """Qt signals mirroring the controller's run lifecycle callbacks."""

    started = pyqtSignal()
    stopped = pyqtSignal()
    errored = pyqtSignal(str)

    def attach(self, controller) -> None:
        """Wire a controller's callbacks to emit these signals."""
        controller.on_started = self.started.emit
        controller.on_stopped = self.stopped.emit
        controller.on_error = lambda error: self.errored.emit(str(error))
