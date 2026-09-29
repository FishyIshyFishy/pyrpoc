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
    #: A panel type was chosen from Add, by registry key.
    panel_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.panels_menu = QMenu("Panels", self)
        self.addMenu(self.panels_menu)
        # Parented to the menu bar rather than to Panels, because
        # populate_panels_menu clears that menu every time the open panels
        # change and would otherwise be re-adding a submenu it had orphaned.
        self.add_menu = QMenu("Add", self)

        self.style_menu = QMenu("&Style", self)
        self.addMenu(self.style_menu)
        self._style_actions: dict[str, QAction] = {}
        self._style_group = QActionGroup(self)
        self._style_group.setExclusive(True)

    def populate_add_menu(self, entries: list[tuple[str, str]]) -> None:
        """What can be added, as (key, label). Fixed for the session.

        A flat list, browsed by hovering Add the way a colour is browsed under
        Style. Grouping panel types is a decision nobody has needed to make
        yet, and a submenu per group over four entries would cost two hovers to
        reach every one of them.
        """
        self.add_menu.clear()
        for key, label in entries:
            action = QAction(label, self.add_menu)
            action.triggered.connect(lambda _checked=False, k=key: self.panel_requested.emit(k))
            self.add_menu.addAction(action)

    def populate_panels_menu(self, docks: list, panel_actions: list[QAction] | None = None) -> None:
        """The fixed panels, then Add, then one entry per panel that was added.

        Unchecking one of the fixed docks hides it; unchecking an added one
        destroys it. The rule is the only thing separating the two, so the
        entries below it are kept under Add rather than fenced off from it --
        what they are is "what Add has produced".
        """
        self.panels_menu.clear()
        for dock in docks:
            self.panels_menu.addAction(dock.toggleViewAction())
        self.panels_menu.addSeparator()
        self.panels_menu.addMenu(self.add_menu)
        for action in panel_actions or []:
            if sip.isdeleted(action):
                continue
            if not action.isCheckable():
                action.setCheckable(True)
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
