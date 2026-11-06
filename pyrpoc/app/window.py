"""The dock manager: three fixed panels, one dock per added data panel, and
the device panels ``DeviceDocks`` keeps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import PyQt6Ads as qtads
from PyQt6.QtCore import QByteArray, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from pyrpoc.app.application import Application
from pyrpoc.app.device_docks import DeviceDocks
from pyrpoc.app.menubar import MainMenuBar
from pyrpoc.app.theme.manager import ThemeController
from pyrpoc.core.panels import Panel
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.registry import grouped
from pyrpoc.engine.acquisition.panel import AcquisitionPanel
from pyrpoc.engine.data_library.panel import DataLibraryPanel
from pyrpoc.engine.device_inventory.panel import DevicesPanel

# Importing the plugins registers every device, program and panel.
from pyrpoc.plugins import data_panel_registry, device_registry
from pyrpoc.utils.widgets.data_panel import DataPanel
from pyrpoc.utils.widgets.device_panel import has_panel

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
    # Screen state worth saving changed: a panel came or went, or one changed
    # its own state.
    screen_changed = pyqtSignal()

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
        self.fixed_panels: dict[DockKey, Panel] = {}
        self.panel_docks: dict[DataPanel, qtads.CDockWidget] = {}
        self.panel_actions: dict[DataPanel, QAction] = {}
        # Which instance of its type each panel is, to tell same-kind docks apart.
        self.panel_ordinals: dict[DataPanel, int] = {}

        self.device_docks = DeviceDocks(self.app.inventory, self.dock_manager, self)
        self.device_docks.actions_changed.connect(self.refresh_panels_menu)

        self.menubar = MainMenuBar(self)
        self.populate_add_menus()
        self.menubar.panel_requested.connect(self.add_panel_of_type)
        self.menubar.device_requested.connect(self.add_device)
        self.build_panels()

        self.app.library.output_changed.connect(self.on_output_changed)

        layout = QVBoxLayout(self)
        layout.setMenuBar(self.menubar)
        layout.addWidget(self.dock_manager)

        self.refresh_panels_menu()
        self.menubar.populate_style_menu(self.theme_controller.get_saved_mode())
        self.menubar.style_selected.connect(self.set_style)

    def bind_session(self, save_now: Callable[[], None]) -> None:
        """Save the session when the window closes."""
        self.closing.connect(save_now)

    def populate_add_menus(self) -> None:
        """Displays under their headings; misc panels and devices flat."""
        displays: dict[str, list[tuple[str, str]]] = {}
        misc: list[tuple[str, str]] = []
        for heading, keys in grouped(data_panel_registry).items():
            for key in keys:
                cls = data_panel_registry.get(key)
                if cls.misc:
                    misc.append((key, cls.display_name))
                else:
                    displays.setdefault(heading, []).append((key, cls.display_name))
        devices = sorted(
            ((key, device_registry.get(key).display_name) for key in device_registry.keys()),
            key=lambda entry: entry[1],
        )
        self.menubar.populate_add_menus(displays, devices, misc)

    def add_device(self, key: str) -> None:
        """Add one from the menu, opening its panel straight away."""
        device = self.app.inventory.add(key)
        if has_panel(device):
            self.device_docks.open(device)

    def build_panels(self) -> None:
        app = self.app
        self.fixed_panels = {
            DockKey.ACQUISITION: AcquisitionPanel(app.runs, app.inputs),
            DockKey.DEVICES: DevicesPanel(app.inventory),
            DockKey.DATA: DataLibraryPanel(app.library, lambda: app.runs.save.folder),
        }
        for panel in self.fixed_panels.values():
            panel.state_changed.connect(self.screen_changed.emit)
        first: qtads.CDockWidget | None = None
        for spec in PANELS:
            panel = self.fixed_panels[spec.key]
            dock = self.add_dock(spec.title, panel, spec.object_name, tab_with=first)
            self.dock_by_key[spec.key] = dock
            if first is None:
                first = dock
        # Every job starts here; the tab added last would otherwise be the one shown.
        self.dock_by_key[DockKey.ACQUISITION].setAsCurrentTab()

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

    @property
    def panels(self) -> list[DataPanel]:
        """The added panels, in the order they were added."""
        return list(self.panel_docks)

    def add_panel_of_type(self, key: str) -> None:
        """Add one, from the menu."""
        self.add_panel(
            data_panel_registry.get(key)(self.app.library, self.app.inputs, self.app.requests)
        )

    def add_panel(self, panel: DataPanel) -> None:
        self.add_panel_dock(panel)
        self.refresh_panels_menu()
        self.screen_changed.emit()

    def remove_panel(self, panel: DataPanel) -> None:
        panel.detach()
        self.remove_panel_dock(panel)
        self.refresh_panels_menu()
        self.screen_changed.emit()

    def fixed_panel_states(self) -> dict[str, dict[str, Any]]:
        """Each fixed panel's own screen state, by dock key."""
        return {
            key.value: panel.export_persistence_state() for key, panel in self.fixed_panels.items()
        }

    def restore_fixed_panel_states(self, states: dict[str, dict[str, Any]]) -> None:
        """Give each fixed panel its saved state; one with none gets an empty state."""
        for key, panel in self.fixed_panels.items():
            panel.import_persistence_state(states.get(key.value, {}))

    def clear_panels(self) -> None:
        for panel in self.panels:
            self.remove_panel(panel)

    def on_output_changed(self, program_output: ProgramOutput) -> None:
        """Refresh every panel showing this program output."""
        # An added panel exists exactly as long as its dock, so there is no unseen panel to skip.
        for panel in self.panels:
            if panel.program_output() is program_output:
                panel.refresh()

    def assign_ordinal(self, panel: DataPanel) -> None:
        """Give ``panel`` the lowest number its type has free."""
        # Assigned once, so a dock never renumbers itself when an earlier one closes.
        taken = {
            number
            for other, number in self.panel_ordinals.items()
            if other.registry_key == panel.registry_key
        }
        number = 1
        while number in taken:
            number += 1
        self.panel_ordinals[panel] = number

    def panel_title(self, panel: DataPanel) -> str:
        if panel.user_label:
            return panel.user_label
        number = self.panel_ordinals[panel]
        return panel.display_name if number == 1 else f"{panel.display_name} {number}"

    @staticmethod
    def panel_object_name(panel: DataPanel) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in panel.instance_id)
        # Saved layouts name added docks with this prefix; see PANELS above.
        return f"dock.view.{safe}"

    def add_panel_dock(self, panel: DataPanel) -> None:
        self.assign_ordinal(panel)
        dock = qtads.CDockWidget(self.panel_title(panel))
        dock.setObjectName(self.panel_object_name(panel))
        dock.setWidget(panel)
        self.dock_beside_displays(dock)
        self.panel_docks[panel] = dock

        action = QAction(self.panel_title(panel), self)
        action.setCheckable(True)
        action.setChecked(True)
        action.toggled.connect(lambda checked, p=panel: self.on_panel_toggled(p, checked))
        self.panel_actions[panel] = action
        dock.closed.connect(lambda *_args, p=panel: self.discard_panel(p))

    def dock_beside_displays(self, dock: qtads.CDockWidget) -> None:
        """Add ``dock`` as a tab beside the added panels still in the window, or to the right
        of the fixed panels when there are none."""
        # Inside the window rather than floating: a loose window covers the panels it is opened
        # to sit beside. It can still be dragged out.
        docked = [each for each in self.panel_docks.values() if not each.isFloating()]
        area = docked[-1].dockAreaWidget() if docked else None
        if area is None:
            added = self.dock_manager.addDockWidget(qtads.DockWidgetArea.RightDockWidgetArea, dock)
            # Displays are what a user looks at, so they get a little more than half the width.
            width = self.dock_manager.width()
            self.dock_manager.setSplitterSizes(added, [width * 9 // 20, width * 11 // 20])
        else:
            self.dock_manager.addDockWidgetTabToArea(dock, area)

    def remove_panel_dock(self, panel: DataPanel) -> None:
        dock = self.panel_docks.pop(panel)
        action = self.panel_actions.pop(panel)
        del self.panel_ordinals[panel]

        self.menubar.panels_menu.removeAction(action)
        action.deleteLater()

        self.dock_manager.removeDockWidget(dock)
        dock.takeWidget()
        panel.setParent(None)
        dock.deleteLater()

    def discard_panel(self, panel: DataPanel) -> None:
        """Drop an added panel, from its tab's close button or its menu entry."""
        # Deferred, because both arrive in a signal from the thing about to be deleted. A close
        # and an uncheck can both arrive for one gesture; the second finds the panel already gone.
        if self.restoring_layout or self.shutting_down:
            return
        if panel not in self.panel_docks:
            return
        QTimer.singleShot(0, lambda p=panel: self.remove_panel(p))

    def on_panel_toggled(self, panel: DataPanel, visible: bool) -> None:
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
        self.restoring_layout = self.device_docks.frozen = True
        try:
            self.dock_manager.restoreState(data)
        finally:
            self.restoring_layout = self.device_docks.frozen = False
        self.refresh_panels_menu()

    def refresh_panels_menu(self) -> None:
        actions = self.panel_actions
        self.menubar.populate_panels_menu(
            list(self.dock_by_key.values()),
            [action for panel, action in actions.items() if not panel.misc],
            list(self.device_docks.actions.values()),
            [action for panel, action in actions.items() if panel.misc],
        )

    def set_style(self, theme_mode: str) -> None:
        self.menubar.set_active_style(self.theme_controller.apply(theme_mode, persist=True))

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self.shutting_down = self.device_docks.frozen = True
        self.closing.emit()
        super().closeEvent(event)
