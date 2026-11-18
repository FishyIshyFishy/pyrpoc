"""Where a new dock goes: as a loose window once the saved layout is restored, and before
that as a tab holding a stand-in until the layout is in."""

from __future__ import annotations

import PyQt6Ads as qtads
from PyQt6.QtWidgets import QWidget


class DockPlacement:
    """Adds docks to the dock manager, holding back their panels until the saved layout is
    restored."""

    def __init__(self, dock_manager: qtads.CDockManager):
        self.dock_manager = dock_manager
        self.layout_restored = False
        # Each held dock with its stand-in and its panel.
        self.held: list[tuple[qtads.CDockWidget, QWidget, QWidget]] = []

    def add(self, dock: qtads.CDockWidget, panel: QWidget) -> None:
        if self.layout_restored:
            dock.setWidget(panel)
            self.dock_manager.addDockWidgetFloating(dock)
            return
        # Moving a panel re-applies the app stylesheet to every widget in it, and restoring
        # the layout moves every dock, so docks move empty and each panel moves in once.
        # A tab, not a loose window, so nothing appears on screen before the layout is in.
        stand_in = QWidget()
        dock.setWidget(stand_in)
        self.held.append((dock, stand_in, panel))
        self.dock_manager.addDockWidgetTab(qtads.DockWidgetArea.LeftDockWidgetArea, dock)

    def release(self) -> None:
        """Put each held panel in its dock, and add docks as loose windows from now on."""
        for dock, stand_in, panel in self.held:
            dock.setWidget(panel)
            stand_in.deleteLater()
        self.held.clear()
        self.layout_restored = True
