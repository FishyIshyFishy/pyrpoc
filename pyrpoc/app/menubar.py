from __future__ import annotations

import PyQt6Ads as qtads
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QColor, QIcon, QPixmap
from PyQt6.QtWidgets import QMenu, QMenuBar, QWidget

from pyrpoc.app.theme.mfruit import ACCENTS
from pyrpoc.app.theme.palette import Mode, Theme


class MainMenuBar(QMenuBar):
    mode_selected = pyqtSignal(Mode)
    accent_selected = pyqtSignal(str)
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
        self._mode_actions: dict[Mode, QAction] = {}
        self._accent_actions: dict[str, QAction] = {}

    def populate_add_menu(self, groups: dict[str, list[tuple[str, str]]]) -> None:
        """What can be added, as (key, label) under each heading, one submenu
        per heading. Fixed for the session."""
        self.add_menu.clear()
        for heading, entries in groups.items():
            group_menu = QMenu(heading, self.add_menu)
            self.add_menu.addMenu(group_menu)
            for key, label in entries:
                action = QAction(label, group_menu)
                action.triggered.connect(lambda _checked=False, k=key: self.panel_requested.emit(k))
                group_menu.addAction(action)

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

    def populate_style_menu(self, selected: Theme) -> None:
        self.style_menu.clear()
        self._mode_actions.clear()
        self._accent_actions.clear()
        mode_group = QActionGroup(self.style_menu)
        for mode in Mode:
            action = QAction(mode.value.title(), self.style_menu)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked, m=mode: self.mode_selected.emit(m))
            mode_group.addAction(action)
            self.style_menu.addAction(action)
            self._mode_actions[mode] = action
        self.style_menu.addSeparator()
        accent_group = QActionGroup(self.style_menu)
        for name, hex_color in ACCENTS.items():
            swatch = QPixmap(14, 14)
            swatch.fill(QColor(hex_color))
            action = QAction(QIcon(swatch), name, self.style_menu)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked, n=name: self.accent_selected.emit(n))
            accent_group.addAction(action)
            self.style_menu.addAction(action)
            self._accent_actions[name] = action
        self.set_active_style(selected)

    def set_active_style(self, selected: Theme) -> None:
        for mode, action in self._mode_actions.items():
            action.setChecked(mode is selected.mode)
        for name, action in self._accent_actions.items():
            action.setChecked(name == selected.accent)
