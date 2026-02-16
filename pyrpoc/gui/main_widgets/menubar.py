from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMenu, QMenuBar


class MainMenuBar(QMenuBar):
    style_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.file_menu = QMenu("&File", self)
        self.file_menu.addAction(QAction("New", self))
        self.file_menu.addAction(QAction("Open...", self))
        self.file_menu.addAction(QAction("Save", self))
        self.file_menu.addAction(QAction("Save As...", self))
        self.file_menu.addSeparator()
        self.file_menu.addAction(QAction("Exit", self))
        self.addMenu(self.file_menu)

        self.view_menu = QMenu("View", self)
        self.addMenu(self.view_menu)

        self.prefs_menu = QMenu("&Preferences", self)
        self.prefs_menu.addAction(QAction("Application Settings", self))
        self.prefs_menu.addAction(QAction("Keyboard Shortcuts", self))
        self.addMenu(self.prefs_menu)

        self.style_menu = QMenu("&Style", self)
        self.addMenu(self.style_menu)

    def populate_view_menu(self, docks: list) -> None:
        self.view_menu.clear()
        for dock in docks:
            self.view_menu.addAction(dock.toggleViewAction())

    def populate_style_menu(self, themes: list[str]) -> None:
        self.style_menu.clear()
        for theme in themes:
            action = QAction(theme.capitalize(), self.style_menu)
            action.triggered.connect(lambda checked, t=theme: self.style_selected.emit(t))
            self.style_menu.addAction(action)
