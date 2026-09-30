"""Saving the workspace shortly after anything changes, and the explicit
restore and reset actions."""

from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, QTimer

from pyrpoc.plugins.programs import program_registry

from ..gui.window import MainWindow
from ..model.application import Application
from .file import SaveState, WorkspaceFile
from .restore import apply, capture, seed_defaults

log = logging.getLogger(__name__)


class Autosave(QObject):
    """Debounced save on any state change, plus explicit save/reset actions."""

    def __init__(self, app: Application, window: MainWindow, store: WorkspaceFile, parent: QObject):
        super().__init__(parent)
        self.app = app
        self.window = window
        self.store = store
        self.suspended = False

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.save_now)

        app.state_changed.connect(self.schedule)
        window.panels_changed.connect(self.schedule)
        app.devices_changed.connect(self.schedule)

    def schedule(self) -> None:
        if not self.suspended:
            self.timer.start()

    def save_now(self) -> None:
        if self.suspended:
            return
        # A file boundary: a failed autosave is logged and must never
        # interrupt an experiment.
        try:
            self.store.save(capture(self.app, self.window))
        except OSError:
            log.warning("could not save the workspace to %s", self.store.path, exc_info=True)

    def restore(self) -> None:
        self.suspended = True
        try:
            apply(self.store.load(), self.app, self.window)
            seed_defaults(self.app)
        finally:
            self.suspended = False
        self.save_now()

    def reset(self) -> None:
        self.suspended = True
        try:
            self.window.clear_panels()
            self.app.clear_devices()
            self.app.blocks.clear()
            self.app.set_save(name=SaveState().name, directory="", enabled=False)
            seed_defaults(self.app)
            self.app.select_program(program_registry.keys()[0])
        finally:
            self.suspended = False
        self.save_now()
