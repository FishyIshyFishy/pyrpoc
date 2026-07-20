"""The main window: the old GUI's exact shell (QWidget + menubar + ADS docks).

A plain QWidget hosting a menubar and the dock manager — not a QMainWindow — so the
tab bar and chrome render identically to the previous GUI. Acquisition / Instruments /
Displays are tabbed tool docks; the routine editor is a fourth tab hidden until Ctrl+R;
displays open as docks on the right.
"""

from __future__ import annotations

import PyQt6Ads as qtads
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QCloseEvent, QKeySequence
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from pyrpoc_next.core import Controller
from pyrpoc_next.gui.bridge import RunBridge
from pyrpoc_next.gui.menubar import MainMenuBar
from pyrpoc_next.gui.panels import AcquisitionPanel, DisplaysPanel, InstrumentManagerWidget, RoutineEditor

qtads.CDockManager.setConfigFlag(qtads.CDockManager.eConfigFlag.DisableTabTextEliding, True)
qtads.CDockManager.setConfigFlag(qtads.CDockManager.eConfigFlag.OpaqueSplitterResize, False)


class MainWindow(QWidget):
    """Top-level window wiring the dock panels to the core controller."""

    closing = pyqtSignal()

    def __init__(self, controller: Controller):
        super().__init__()
        self.setWindowTitle("pyrpoc")
        self.controller = controller
        self.bridge = RunBridge()
        self.bridge.attach(controller)

        self.dock_manager = qtads.CDockManager(self)
        self.menubar = MainMenuBar(self)

        self.acquisition = AcquisitionPanel(controller, self.bridge)
        self.instruments = InstrumentManagerWidget(controller.state)
        self.displays = DisplaysPanel(controller.state, self.open_display_dock, self.close_dock)
        self.routine_editor = RoutineEditor(controller)
        self.routine_editor.changed.connect(self.acquisition.rebuild)

        self.add_tool_dock("Acquisition", self.acquisition, first=True)
        self.add_tool_dock("Instruments", self.instruments)
        self.add_tool_dock("Displays", self.displays)
        self.routine_dock = self.add_tool_dock("Routine", self.routine_editor)
        self.routine_dock.toggleView(False)  # hidden until Ctrl+R
        self.acquisition.rebuild()  # pick up the block the editor seeded

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setMenuBar(self.menubar)
        layout.addWidget(self.dock_manager)

        self.refresh_view_menu()
        self.menubar.populate_style_menu("dark")
        self.menubar.style_selected.connect(self.on_style_selected)
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

    def refresh_view_menu(self) -> None:
        self.menubar.populate_view_menu(list(self.dock_manager.dockWidgetsMap().values()))
        self.routine_dock.toggleViewAction().setShortcut(QKeySequence("Ctrl+R"))

    def on_style_selected(self, mode: str) -> None:
        from PyQt6.QtWidgets import QApplication

        from pyrpoc_next.gui.styles.theme import apply_dark_theme

        app = QApplication.instance()
        if app is not None:
            apply_dark_theme(app)  # only breeze dark is available for now

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt override)
        self.closing.emit()
        super().closeEvent(event)
