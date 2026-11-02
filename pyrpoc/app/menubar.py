from __future__ import annotations

from collections.abc import Callable

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
    # A display or misc panel type was chosen to add, by registry key.
    panel_requested = pyqtSignal(str)
    # A device type was chosen to add, by registry key.
    device_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget):
        super().__init__(parent)

        self.panels_menu = QMenu("Panels", self)
        self.addMenu(self.panels_menu)
        # Parented to the bar, not to Panels: populate_panels_menu clears Panels
        # on every change and would otherwise orphan these submenus.
        self.add_display_menu = QMenu("Add Display", self)
        self.add_device_menu = QMenu("Add Device", self)
        self.add_misc_menu = QMenu("Add Misc. Panel", self)

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
        """What can be added, as (key, label): displays in one submenu per
        heading, devices and misc panels flat. Fixed for the session."""
        self.add_display_menu.clear()
        for heading, entries in displays.items():
            group_menu = QMenu(heading, self.add_display_menu)
            self.add_display_menu.addMenu(group_menu)
            add_entries(group_menu, entries, self.panel_requested.emit)
        self.add_device_menu.clear()
        add_entries(self.add_device_menu, devices, self.device_requested.emit)
        self.add_misc_menu.clear()
        add_entries(self.add_misc_menu, misc, self.panel_requested.emit)

    def populate_panels_menu(
        self,
        docks: list[qtads.CDockWidget],
        display_actions: list[QAction],
        device_actions: list[QAction],
        misc_actions: list[QAction],
    ) -> None:
        """The fixed panels, then a section each for displays, devices and misc
        panels: its Add submenu, then one entry per panel. Unchecking a fixed
        dock hides it; unchecking an added one destroys it, and unchecking a
        device closes its panel but keeps the device."""
        self.panels_menu.clear()
        for dock in docks:
            self.panels_menu.addAction(dock.toggleViewAction())
        for add_menu, actions in (
            (self.add_display_menu, display_actions),
            (self.add_device_menu, device_actions),
            (self.add_misc_menu, misc_actions),
        ):
            self.panels_menu.addSeparator()
            self.panels_menu.addMenu(add_menu)
            for action in actions:
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


def add_entries(menu: QMenu, entries: list[tuple[str, str]], chosen: Callable[[str], None]) -> None:
    for key, label in entries:
        action = QAction(label, menu)
        action.triggered.connect(lambda _checked=False, k=key: chosen(k))
        menu.addAction(action)
