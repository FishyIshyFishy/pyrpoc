"""The app's working state, built off-screen with the simulated programs registered."""

from __future__ import annotations

import os

from PyQt6.QtWidgets import QApplication

from pyrpoc.app.application import Application
from pyrpoc.core.program import program_registry

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# Kept here: a QApplication with no Python reference left is destroyed with its children.
_APP: list[QApplication] = []


def qt_app() -> QApplication:
    if not _APP:
        _APP.append(QApplication([]))
    return _APP[0]


def application() -> Application:
    """A fresh application with the simulated programs registered."""
    qt_app()
    if "simulation" not in program_registry.entries:
        from pyrpoc.plugins.programs.simulated import register

        register()
    return Application()
