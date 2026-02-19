from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QMenu, QMenuBar


class MainMenuBar(QMenuBar):
    style_selected = pyqtSignal(str)
    new_requested = pyqtSignal()
    open_requested = pyqtSignal()
    save_requested = pyqtSignal()
    save_as_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.file_menu = QMenu("&File", self)
        self.new_action = QAction("New", self)
        self.new_action.triggered.connect(self.new_requested.emit)
        self.file_menu.addAction(self.new_action)

        self.open_action = QAction("Open...", self)
        self.open_action.triggered.connect(self.open_requested.emit)
        self.file_menu.addAction(self.open_action)

        self.save_action = QAction("Save", self)
        self.save_action.triggered.connect(self.save_requested.emit)
        self.file_menu.addAction(self.save_action)

        self.save_as_action = QAction("Save As...", self)
        self.save_as_action.triggered.connect(self.save_as_requested.emit)
        self.file_menu.addAction(self.save_as_action)

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
        self._style_actions: dict[str, QAction] = {}
        self._style_group = QActionGroup(self)
        self._style_group.setExclusive(True)

    def populate_view_menu(self, docks: list) -> None:
        self.view_menu.clear()
        for dock in docks:
            self.view_menu.addAction(dock.toggleViewAction())

    def populate_style_menu(self, selected_mode: str) -> None:
        self.style_menu.clear()
        self._style_actions.clear()
        mode_items = [
            ("system", "Follow System"),
            ("dark", "Dark"),
            ("light", "Light"),
        ]
        for mode, label in mode_items:
            action = QAction(label, self.style_menu)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, m=mode: self.style_selected.emit(m))
            self._style_group.addAction(action)
            self.style_menu.addAction(action)
            self._style_actions[mode] = action
        self.set_active_style(selected_mode)

    def set_active_style(self, selected_mode: str) -> None:
        for mode, action in self._style_actions.items():
            action.setChecked(mode == selected_mode)
