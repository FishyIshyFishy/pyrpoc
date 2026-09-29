"""The dock manager: three fixed panels plus one dock per added panel.

The fixed panels hide when unchecked in the menu; an added panel is destroyed,
so it exists exactly as long as its dock does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import PyQt6Ads as qtads
from PyQt6.QtCore import QByteArray, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from pyrpoc.src.panels import (
    AcquisitionPanel,
    DataLibraryPanel,
    DatasetPanel,
    DevicesPanel,
    panel_registry,
)

from .application import Application
from .menubar import MainMenuBar
from .theme.manager import ThemeController

qtads.CDockManager.setConfigFlag(qtads.CDockManager.eConfigFlag.DisableTabTextEliding, True)
qtads.CDockManager.setConfigFlag(qtads.CDockManager.eConfigFlag.OpaqueSplitterResize, False)


class DockKey(str, Enum):
    ACQUISITION = "acquisition"
    DEVICES = "devices"
    DATA = "data"


@dataclass(frozen=True)
class DockSpec:
    key: DockKey
    title: str
    object_name: str


PANELS = [
    DockSpec(DockKey.ACQUISITION, "Acquisition", "dock.acquisition"),
    DockSpec(DockKey.DEVICES, "Devices", "dock.devices"),
    # ADS restores docks by object name, so renaming this strands saved layouts.
    DockSpec(DockKey.DATA, "Data Library", "dock.views"),
]


class MainWindow(QWidget):
    closing = pyqtSignal()

    def __init__(self, app: Application, theme_controller: ThemeController):
        super().__init__()
        self.setWindowTitle("pyrpoc")
        self.app = app
        self.theme_controller = theme_controller

        self.dock_manager = qtads.CDockManager(self)
        self.dock_manager.setStyleSheet("")
        # restoreState() and teardown both close docks, and closing an added
        # panel's dock deletes the panel; these guard the inventory meanwhile.
        self.restoring_layout = False
        self.shutting_down = False
        self.dock_by_key: dict[DockKey, qtads.CDockWidget] = {}
        self.panel_docks: dict[DatasetPanel, qtads.CDockWidget] = {}
        self.panel_actions: dict[DatasetPanel, QAction] = {}
        # Which instance of its type each panel is, to tell same-kind docks apart.
        self.panel_ordinals: dict[DatasetPanel, int] = {}

        self.menubar = MainMenuBar(self)
        self.menubar.populate_add_menu(
            [(key, panel_registry.get(key).display_name) for key in panel_registry.keys()]
        )
        self.menubar.panel_requested.connect(self.add_panel_of_type)
        self.build_panels()

        self.app.panels_changed.connect(self.sync_panel_docks)

        layout = QVBoxLayout(self)
        layout.setMenuBar(self.menubar)
        layout.addWidget(self.dock_manager)

        self.refresh_panels_menu()
        self.menubar.populate_style_menu(self.theme_controller.get_saved_mode())
        self.menubar.style_selected.connect(self.set_style)

    def bind_session(self, save_now: Callable[[], None]) -> None:
        """Save the session when the window closes."""
        self.closing.connect(save_now)

    def build_panels(self) -> None:
        widgets = {
            DockKey.ACQUISITION: AcquisitionPanel(self.app),
            DockKey.DEVICES: DevicesPanel(self.app),
            DockKey.DATA: DataLibraryPanel(self.app),
        }
        first: qtads.CDockWidget | None = None
        for spec in PANELS:
            dock = self.add_dock(spec.title, widgets[spec.key], spec.object_name, tab_with=first)
            self.dock_by_key[spec.key] = dock
            if first is None:
                first = dock

    def add_dock(
        self,
        title: str,
        widget: QWidget,
        object_name: str,
        tab_with: qtads.CDockWidget | None,
    ) -> qtads.CDockWidget:
        dock = qtads.CDockWidget(title)
        # ADS keys save/restore by the object name at add time, so set it first.
        dock.setObjectName(object_name)
        dock.setWidget(widget)
        area = qtads.DockWidgetArea.LeftDockWidgetArea
        if tab_with is None:
            self.dock_manager.addDockWidget(area, dock)
        else:
            self.dock_manager.addDockWidgetTab(area, dock)
        return dock

    def add_panel_of_type(self, key: str) -> None:
        """Add one, from the menu. The dock follows from panels_changed."""
        self.app.add_panel(panel_registry.get(key)(self.app.library))

    def sync_panel_docks(self) -> None:
        for panel in list(self.panel_docks):
            if panel not in self.app.panels:
                self.remove_panel_dock(panel)
        for panel in self.app.panels:
            if panel not in self.panel_docks:
                self.add_panel_dock(panel)
        self.refresh_panels_menu()

    def assign_ordinal(self, panel: DatasetPanel) -> None:
        """The lowest number this type has free, assigned once so a dock never
        renumbers itself when an earlier one closes."""
        taken = {
            number
            for other, number in self.panel_ordinals.items()
            if other.type_key == panel.type_key
        }
        number = 1
        while number in taken:
            number += 1
        self.panel_ordinals[panel] = number

    def panel_title(self, panel: DatasetPanel) -> str:
        if panel.user_label:
            return panel.user_label
        number = self.panel_ordinals[panel]
        return panel.display_name if number == 1 else f"{panel.display_name} {number}"

    @staticmethod
    def panel_object_name(panel: DatasetPanel) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in panel.instance_id)
        # Saved layouts name added docks with this prefix; see PANELS above.
        return f"dock.view.{safe}"

    def add_panel_dock(self, panel: DatasetPanel) -> None:
        self.assign_ordinal(panel)
        dock = qtads.CDockWidget(self.panel_title(panel))
        dock.setObjectName(self.panel_object_name(panel))
        dock.setWidget(panel)
        self.dock_manager.addDockWidget(qtads.DockWidgetArea.RightDockWidgetArea, dock)
        self.panel_docks[panel] = dock

        action = QAction(self.panel_title(panel), self)
        action.setCheckable(True)
        action.setChecked(True)
        action.toggled.connect(lambda checked, p=panel: self.on_panel_toggled(p, checked))
        self.panel_actions[panel] = action
        dock.closed.connect(lambda *_args, p=panel: self.discard_panel(p))

    def remove_panel_dock(self, panel: DatasetPanel) -> None:
        dock = self.panel_docks.pop(panel)
        action = self.panel_actions.pop(panel)
        del self.panel_ordinals[panel]

        self.menubar.panels_menu.removeAction(action)
        action.deleteLater()

        self.dock_manager.removeDockWidget(dock)
        dock.takeWidget()
        panel.setParent(None)
        dock.deleteLater()

    def discard_panel(self, panel: DatasetPanel) -> None:
        """Drop an added panel, from its tab's close button or its menu entry.

        Deferred, because both arrive in a signal from the thing about to be
        deleted. The membership check is because a close and an uncheck can
        both arrive for one gesture; the second finds the panel already gone.
        """
        if self.restoring_layout or self.shutting_down:
            return
        if panel not in self.app.panels:
            return
        QTimer.singleShot(0, lambda p=panel: self.app.remove_panel(p))

    def on_panel_toggled(self, panel: DatasetPanel, visible: bool) -> None:
        """Unchecking an added panel deletes it; there is nothing to re-check."""
        if not visible:
            self.discard_panel(panel)

    def save_dock_layout(self) -> str | None:
        state = self.dock_manager.saveState()
        if state.isEmpty():
            return None
        return state.toBase64().data().decode("ascii")

    def restore_dock_layout(self, layout_base64: str | None) -> None:
        if not layout_base64:
            return
        data = QByteArray.fromBase64(layout_base64.encode("ascii"))
        if data.isEmpty():
            return
        self.restoring_layout = True
        try:
            self.dock_manager.restoreState(data)
        finally:
            self.restoring_layout = False
        self.refresh_panels_menu()

    def refresh_panels_menu(self) -> None:
        self.menubar.populate_panels_menu(
            list(self.dock_by_key.values()), list(self.panel_actions.values())
        )

    def set_style(self, theme_mode: str) -> None:
        self.menubar.set_active_style(self.theme_controller.apply(theme_mode, persist=True))

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self.shutting_down = True
        self.closing.emit()
        super().closeEvent(event)
