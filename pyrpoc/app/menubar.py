from __future__ import annotations

import PyQt6Ads as qtads
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QMenu, QMenuBar, QWidget

from pyrpoc.app.theme.manager import available_breeze_themes

_STYLE_COLOR_GROUPS = ["blue", "red", "green", "purple", "cyan", "pink"]
_STYLE_VARIANTS = [
    ("light-{c}", "Light"),
    ("light-{c}-alt", "Light Alt"),
    ("dark-{c}", "Dark"),
    ("dark-{c}-alt", "Dark Alt"),
]


class MainMenuBar(QMenuBar):
    style_selected = pyqtSignal(str)
    # A panel type was chosen from Add, by registry key.
    panel_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget):
        super().__init__(parent)

        self.panels_menu = QMenu("Panels", self)
        self.addMenu(self.panels_menu)
        # Parented to the bar, not to Panels: populate_panels_menu clears Panels
        # on every change and would otherwise orphan this submenu.
        self.add_menu = QMenu("Add", self)

        self.style_menu = QMenu("&Style", self)
        self.addMenu(self.style_menu)
        self._style_actions: dict[str, QAction] = {}
        self._style_group = QActionGroup(self)
        self._style_group.setExclusive(True)

    def populate_add_menu(self, entries: list[tuple[str, str]]) -> None:
        """What can be added, as (key, label). Fixed for the session."""
        self.add_menu.clear()
        for key, label in entries:
            action = QAction(label, self.add_menu)
            action.triggered.connect(lambda _checked=False, k=key: self.panel_requested.emit(k))
            self.add_menu.addAction(action)

    def populate_panels_menu(
        self, docks: list[qtads.CDockWidget], panel_actions: list[QAction]
    ) -> None:
        """The fixed panels, a rule, then Add and one entry per added panel.
        Unchecking a fixed dock hides it; unchecking an added one destroys it."""
        self.panels_menu.clear()
        for dock in docks:
            self.panels_menu.addAction(dock.toggleViewAction())
        self.panels_menu.addSeparator()
        self.panels_menu.addMenu(self.add_menu)
        for action in panel_actions:
            self.panels_menu.addAction(action)

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
