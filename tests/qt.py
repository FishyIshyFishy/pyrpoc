"""The app's working state, built off-screen with the simulated programs registered."""

from __future__ import annotations

import os

from PyQt6.QtWidgets import QApplication

from pyrpoc.app.application import Application
from pyrpoc.core.program import program_registry

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def qt_app() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def application() -> Application:
    """A fresh application with the simulated programs registered."""
    qt_app()
    if "simulation" not in program_registry.entries:
        from pyrpoc.plugins.programs.simulated import register

        register()
    return Application()
