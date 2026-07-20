"""The main window: the old GUI's dock layout, re-wired to the new core.

Acquisition / Instruments / Displays are tabbed tool docks (like before). The routine
editor is a fourth tab, hidden until Ctrl+R. Displays open as docks on the right.
"""

from __future__ import annotations

import PyQt6Ads as qtads
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMainWindow, QWidget

from pyrpoc_next.core import Controller
from pyrpoc_next.gui.bridge import RunBridge
from pyrpoc_next.gui.panels import AcquisitionPanel, DisplaysPanel, InstrumentsPanel, RoutineEditor


class MainWindow(QMainWindow):
    """Top-level window wiring the dock panels to the core controller."""

    def __init__(self, controller: Controller):
        super().__init__()
        self.setWindowTitle("pyrpoc")
        self.controller = controller
        self.bridge = RunBridge()
        self.bridge.attach(controller)
        self.dock_manager = qtads.CDockManager(self)

        self.acquisition = AcquisitionPanel(controller, self.bridge)
        self.instruments = InstrumentsPanel(controller.state)
        self.displays = DisplaysPanel(controller.state, self.open_display_dock, self.close_dock)
        self.routine_editor = RoutineEditor(controller)
        self.routine_editor.changed.connect(self.acquisition.rebuild)
        self.acquisition.rebuild()  # pick up the block the editor seeded during construction

        self.add_tool_dock("Acquisition", self.acquisition, first=True)
        self.add_tool_dock("Instruments", self.instruments)
        self.add_tool_dock("Displays", self.displays)
        self.routine_dock = self.add_tool_dock("Routine", self.routine_editor)
        self.routine_dock.toggleView(False)  # hidden until Ctrl+R

        self.build_menu()
        self.resize(1400, 850)

    def add_tool_dock(self, title: str, widget: QWidget, first: bool = False):
        dock = qtads.CDockWidget(title)
        dock.setWidget(widget)
        if first:
            self.dock_manager.addDockWidget(qtads.DockWidgetArea.LeftDockWidgetArea, dock)
        else:
            self.dock_manager.addDockWidgetTab(qtads.DockWidgetArea.LeftDockWidgetArea, dock)
        return dock

    def open_display_dock(self, widget: QWidget, title: str):
        dock = qtads.CDockWidget(title)
        dock.setWidget(widget)
        self.dock_manager.addDockWidget(qtads.DockWidgetArea.RightDockWidgetArea, dock)
        return dock

    def close_dock(self, dock) -> None:
        self.dock_manager.removeDockWidget(dock)

    def build_menu(self) -> None:
        view = self.menuBar().addMenu("View")
        for dock in self.dock_manager.dockWidgetsMap().values():
            view.addAction(dock.toggleViewAction())
        routine_action: QAction = self.routine_dock.toggleViewAction()
        routine_action.setShortcut(QKeySequence("Ctrl+R"))
