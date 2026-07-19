"""The main window: docked panels (routine, instruments, displays) over the core.

The only place that wires Qt to the core Controller. Displays open as their own docks.
"""

from __future__ import annotations

import PyQt6Ads as qtads
from PyQt6.QtWidgets import QMainWindow, QWidget

from pyrpoc_next.core import Controller
from pyrpoc_next.gui.bridge import RunBridge
from pyrpoc_next.gui.panels import DisplaysPanel, InstrumentsPanel, RoutinePanel


class MainWindow(QMainWindow):
    """Top-level window hosting the dock manager and tool panels."""

    def __init__(self, controller: Controller):
        super().__init__()
        self.setWindowTitle("pyrpoc")
        self.controller = controller
        self.bridge = RunBridge()
        self.bridge.attach(controller)
        self.dock_manager = qtads.CDockManager(self)

        routine = RoutinePanel(controller, self.bridge)
        instruments = InstrumentsPanel(controller.state)
        displays = DisplaysPanel(controller.state, self.add_display_dock)

        first = self.add_tool_dock("Routine", routine)
        self.add_tool_dock("Instruments", instruments, tab_into=first)
        self.add_tool_dock("Displays", displays, tab_into=first)
        self.resize(1400, 850)

    def add_tool_dock(self, title: str, widget: QWidget, tab_into=None):
        dock = qtads.CDockWidget(title)
        dock.setWidget(widget)
        if tab_into is None:
            return self.dock_manager.addDockWidget(qtads.DockWidgetArea.LeftDockWidgetArea, dock)
        return self.dock_manager.addDockWidgetTab(qtads.DockWidgetArea.LeftDockWidgetArea, dock)

    def add_display_dock(self, widget: QWidget, title: str) -> None:
        dock = qtads.CDockWidget(title)
        dock.setWidget(widget)
        self.dock_manager.addDockWidget(qtads.DockWidgetArea.RightDockWidgetArea, dock)
