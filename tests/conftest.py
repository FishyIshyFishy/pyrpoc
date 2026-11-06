"""Shared test setup: Qt runs off-screen, with one application for the whole session."""

from __future__ import annotations

import os

import pytest
from PyQt6.QtWidgets import QApplication

# Read when the application is made, not on import, so setting it here is in time.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])
