"""Saving the session shortly after anything changes, and the explicit
restore and reset actions."""

from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, QTimer

from pyrpoc.app.application import Application
from pyrpoc.app.window import MainWindow
from pyrpoc.plugins.programs import program_registry

from .file import SaveState, SessionFile
from .restore import apply, capture, seed_defaults

log = logging.getLogger(__name__)


class Autosave(QObject):
    """Debounced save on any state change, plus explicit save/reset actions."""

    def __init__(self, app: Application, window: MainWindow, store: SessionFile, parent: QObject):
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
        app.inventory.changed.connect(self.schedule)

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
            log.warning("could not save the session to %s", self.store.path, exc_info=True)

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
            self.app.inventory.clear()
            self.app.acquisition.blocks.clear()
            self.app.acquisition.set_save(name=SaveState().name, directory="", enabled=False)
            seed_defaults(self.app)
            self.app.acquisition.select_program(program_registry.keys()[0])
        finally:
            self.suspended = False
        self.save_now()
