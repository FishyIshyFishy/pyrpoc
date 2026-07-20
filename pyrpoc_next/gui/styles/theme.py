"""Apply the breeze dark theme (same look as the previous GUI)."""

from __future__ import annotations

from PyQt6.QtCore import QFile, QTextStream
from PyQt6.QtWidgets import QApplication

from pyrpoc_next.gui.styles import breeze_all  # noqa: F401  registers the Qt resource


def dark_stylesheet() -> str:
    file = QFile(":/dark-pink/stylesheet.qss")
    file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text)
    return QTextStream(file).readAll()


def apply_dark_theme(app: QApplication) -> None:
    app.setStyleSheet(dark_stylesheet())
