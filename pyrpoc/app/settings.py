"""The app's own preferences store, kept by Qt between sessions apart from session files."""

from __future__ import annotations

from PyQt6.QtCore import QSettings


def app_settings() -> QSettings:
    return QSettings("pyrpoc", "pyrpoc")
