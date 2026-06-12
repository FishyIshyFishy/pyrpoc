from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QMenu, QMenuBar
from PyQt6 import sip


class MainMenuBar(QMenuBar):
    style_selected = pyqtSignal(str)
    open_themes_folder_requested = pyqtSignal()
    themes_reload_requested = pyqtSignal()
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

    def populate_view_menu(self, docks: list, display_actions: list[QAction] | None = None) -> None:
        self.view_menu.clear()
        for dock in docks:
            self.view_menu.addAction(dock.toggleViewAction())
        if display_actions:
            if docks:
                self.view_menu.addSeparator()
            for action in display_actions:
                if sip.isdeleted(action):
                    continue
                if not action.isCheckable():
                    action.setCheckable(True)
                self.view_menu.addAction(action)

    def populate_style_menu(self, theme_names: list[str], selected_theme: str) -> None:
        self.style_menu.clear()
        for action in self._style_group.actions():
            self._style_group.removeAction(action)
        self._style_actions.clear()
        for name in theme_names:
            action = QAction(name.replace("-", " ").title(), self.style_menu)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, n=name: self.style_selected.emit(n))
            self._style_group.addAction(action)
            self.style_menu.addAction(action)
            self._style_actions[name] = action

        self.style_menu.addSeparator()
        open_folder_action = QAction("Open Themes Folder...", self.style_menu)
        open_folder_action.triggered.connect(self.open_themes_folder_requested.emit)
        self.style_menu.addAction(open_folder_action)
        reload_action = QAction("Reload Themes", self.style_menu)
        reload_action.triggered.connect(self.themes_reload_requested.emit)
        self.style_menu.addAction(reload_action)

        self.set_active_style(selected_theme)

    def set_active_style(self, selected_theme: str) -> None:
        for name, action in self._style_actions.items():
            action.setChecked(name == selected_theme)
