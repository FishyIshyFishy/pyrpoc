from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QMenu, QMenuBar
from PyQt6 import sip

from .theme.manager import available_breeze_themes

_STYLE_COLOR_GROUPS = ["blue", "red", "green", "purple", "cyan", "pink"]
_STYLE_VARIANTS = [
    ("light-{c}", "Light"),
    ("light-{c}-alt", "Light Alt"),
    ("dark-{c}", "Dark"),
    ("dark-{c}-alt", "Dark Alt"),
]


class MainMenuBar(QMenuBar):
    style_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.view_menu = QMenu("View", self)
        self.addMenu(self.view_menu)

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

    def populate_style_menu(self, selected_mode: str) -> None:
        self.style_menu.clear()
        self._style_actions.clear()
        for color in _STYLE_COLOR_GROUPS:
            color_menu = QMenu(color.title(), self.style_menu)
            self.style_menu.addMenu(color_menu)
            for pattern, label in _STYLE_VARIANTS:
                theme = pattern.format(c=color)
                if theme not in available_breeze_themes:
                    continue
                action = QAction(label, color_menu)
                action.setCheckable(True)
                action.triggered.connect(lambda checked, m=theme: self.style_selected.emit(m))
                self._style_group.addAction(action)
                color_menu.addAction(action)
                self._style_actions[theme] = action
        self.set_active_style(selected_mode)

    def set_active_style(self, selected_mode: str) -> None:
        for mode, action in self._style_actions.items():
            action.setChecked(mode == selected_mode)
