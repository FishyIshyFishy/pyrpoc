from __future__ import annotations

from collections.abc import Callable

import PyQt6Ads as qtads
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QMenu, QMenuBar, QWidget

from pyrpoc.app.theme.mcolors import MCOLORS


class MainMenuBar(QMenuBar):
    style_selected = pyqtSignal(str)
    # A display or misc panel type was chosen to add, by registry key.
    panel_requested = pyqtSignal(str)
    # A device type was chosen to add, by registry key.
    device_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget):
        super().__init__(parent)

        self.panels_menu = QMenu("&Panels", self)
        self.addMenu(self.panels_menu)
        self.displays_menu = QMenu("&Displays", self)
        self.addMenu(self.displays_menu)
        self.instruments_menu = QMenu("&Instruments", self)
        self.addMenu(self.instruments_menu)

        self.style_menu = QMenu("&Style", self)
        self.addMenu(self.style_menu)
        self._style_actions: dict[str, QAction] = {}
        self._style_group = QActionGroup(self)
        self._style_group.setExclusive(True)

    def populate_add_menus(
        self,
        displays: dict[str, list[tuple[str, str]]],
        devices: list[tuple[str, str]],
        misc: list[tuple[str, str]],
    ) -> None:
        """What can be added, as (key, label): displays in one submenu per heading
        with the misc panels below them, and instruments flat. Fixed for the session."""
        self.displays_menu.clear()
        for heading, entries in displays.items():
            group_menu = QMenu(heading, self.displays_menu)
            self.displays_menu.addMenu(group_menu)
            add_entries(group_menu, entries, self.panel_requested.emit)
        if misc:
            self.displays_menu.addSeparator()
            add_entries(self.displays_menu, misc, self.panel_requested.emit)
        self.instruments_menu.clear()
        add_entries(self.instruments_menu, devices, self.device_requested.emit)

    def populate_panels_menu(
        self,
        docks: list[qtads.CDockWidget],
        open_actions: list[QAction],
    ) -> None:
        """Fill the Panels menu: the fixed panels, then one entry per open added panel."""
        # Unchecking a fixed dock hides it; unchecking an added one destroys it, and unchecking a
        # device closes its panel but keeps the device.
        self.panels_menu.clear()
        for dock in docks:
            self.panels_menu.addAction(dock.toggleViewAction())
        if open_actions:
            self.panels_menu.addSeparator()
        for action in open_actions:
            self.panels_menu.addAction(action)

    def populate_style_menu(self, selected_mode: str) -> None:
        self.style_menu.clear()
        self._style_actions.clear()
        for color in MCOLORS:
            color_menu = QMenu(color.label, self.style_menu)
            self.style_menu.addMenu(color_menu)
            for mode in ("dark", "light"):
                theme = f"{mode}-{color.key}"
                action = QAction(mode.title(), color_menu)
                action.setCheckable(True)
                action.triggered.connect(lambda checked, m=theme: self.style_selected.emit(m))
                self._style_group.addAction(action)
                color_menu.addAction(action)
                self._style_actions[theme] = action
        self.set_active_style(selected_mode)

    def set_active_style(self, selected_mode: str) -> None:
        for mode, action in self._style_actions.items():
            action.setChecked(mode == selected_mode)


def add_entries(menu: QMenu, entries: list[tuple[str, str]], chosen: Callable[[str], None]) -> None:
    for key, label in entries:
        action = QAction(label, menu)
        action.triggered.connect(lambda _checked=False, k=key: chosen(k))
        menu.addAction(action)
