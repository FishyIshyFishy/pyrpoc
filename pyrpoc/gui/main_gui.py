from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from PyQt6 import sip
import PyQt6Ads as qtads

from pyrpoc.displays.base_display import BaseDisplay
from pyrpoc.displays.display_registry import display_registry
from pyrpoc.gui.main_widgets.acquisition_mgr import AcquisitionManagerWidget
from pyrpoc.gui.main_widgets.display_mgr import DisplayManagerWidget
from pyrpoc.gui.main_widgets.instrument_mgr import InstrumentManagerWidget
from pyrpoc.gui.main_widgets.menubar import MainMenuBar
from pyrpoc.gui.main_widgets.opto_control_mgr import OptoControlManagerWidget
from pyrpoc.gui.styles.theme_manager import ThemeController
from pyrpoc.services.display_service import DisplayService
from pyrpoc.services.instrument_service import InstrumentService
from pyrpoc.services.modality_service import ModalityService
from pyrpoc.services.opto_control_service import OptoControlService


qtads.CDockManager.setConfigFlag(qtads.CDockManager.eConfigFlag.DisableTabTextEliding, True)
qtads.CDockManager.setConfigFlag(qtads.CDockManager.eConfigFlag.OpaqueSplitterResize, False)


class DockKey(str, Enum):
    ACQUISITION = "acquisition"
    INSTRUMENTS = "instruments"
    DISPLAYS = "displays"
    OPTOCONTROLS = "optocontrols"


@dataclass(frozen=True)
class DockSpec:
    key: DockKey
    title: str
    area: qtads.DockWidgetArea


class MainGUI(QWidget):
    closing = pyqtSignal()

    def __init__(
        self,
        instrument_service: InstrumentService,
        modality_service: ModalityService,
        display_service: DisplayService,
        opto_control_service: OptoControlService,
        theme_controller: ThemeController,
    ):
        super().__init__()
        self.setWindowTitle("pyrpoc")

        self.instrument_service = instrument_service
        self.modality_service = modality_service
        self.display_service = display_service
        self.opto_control_service = opto_control_service
        self.theme_controller = theme_controller

        self.dock_manager = qtads.CDockManager(self)
        self.dock_manager.setStyleSheet("")
        self.docks: list[qtads.CDockWidget] = []
        self.dock_by_key: dict[DockKey, qtads.CDockWidget] = {}
        self.display_docks: dict[BaseDisplay, qtads.CDockWidget] = {}
        self.display_dock_actions: dict[BaseDisplay, QAction] = {}
        self.menubar = MainMenuBar(self)
        self._build_default_docks()

        self.display_service.display_added.connect(self._on_display_added)
        self.display_service.display_removed.connect(self._on_display_removed)
        self.display_service.display_changed.connect(self._on_display_changed)

        layout = QVBoxLayout(self)
        layout.setMenuBar(self.menubar)
        layout.addWidget(self.dock_manager)

        self._refresh_view_menu()
        selected_mode = self.theme_controller.get_saved_mode()
        self.menubar.populate_style_menu(selected_mode)
        self.menubar.style_selected.connect(self.set_style)

    def _build_default_docks(self) -> None:
        dock_specs = [
            DockSpec(DockKey.ACQUISITION, "Acquisition Manager", qtads.DockWidgetArea.LeftDockWidgetArea),
            DockSpec(DockKey.INSTRUMENTS, "Instrument Manager", qtads.DockWidgetArea.LeftDockWidgetArea),
            DockSpec(DockKey.DISPLAYS, "Display Manager", qtads.DockWidgetArea.LeftDockWidgetArea),
            DockSpec(DockKey.OPTOCONTROLS, "Opto-Control Manager", qtads.DockWidgetArea.LeftDockWidgetArea),
        ]

        dock_acq = self.add_dock(
            dock_specs[0].title,
            AcquisitionManagerWidget(self.modality_service),
            dock_specs[0].area,
        )
        dock_acq.setObjectName("dock.acquisition")
        self.dock_by_key[dock_specs[0].key] = dock_acq

        dock_instruments = self.add_dock(
            dock_specs[1].title,
            InstrumentManagerWidget(self.instrument_service),
            dock_specs[1].area,
            tab_with=dock_acq,
        )
        dock_instruments.setObjectName("dock.instruments")
        self.dock_by_key[dock_specs[1].key] = dock_instruments

        dock_displays = self.add_dock(
            dock_specs[2].title,
            DisplayManagerWidget(self.display_service, self.modality_service),
            dock_specs[2].area,
            tab_with=dock_acq,
        )
        dock_displays.setObjectName("dock.displays")
        self.dock_by_key[dock_specs[2].key] = dock_displays

        dock_opto = self.add_dock(
            dock_specs[3].title,
            OptoControlManagerWidget(
                self.opto_control_service,
                self.modality_service,
                self.display_service,
            ),
            dock_specs[3].area,
            tab_with=dock_acq,
        )
        dock_opto.setObjectName("dock.optocontrols")
        self.dock_by_key[dock_specs[3].key] = dock_opto

    def add_dock(
        self,
        title: str,
        widget: QWidget,
        area: qtads.DockWidgetArea,
        tab_with: qtads.CDockWidget | None = None,
    ) -> qtads.CDockWidget:
        dock = qtads.CDockWidget(title)
        dock.setWidget(widget)
        self.docks.append(dock)

        if tab_with is None:
            self.dock_manager.addDockWidget(area, dock)
        else:
            self.dock_manager.addDockWidgetTab(area, dock)

        return dock

    def set_style(self, theme_mode: str) -> None:
        applied = self.theme_controller.apply(theme_mode)
        self.menubar.set_active_style(applied)

    def _refresh_view_menu(self) -> None:
        for display in list(self.display_dock_actions):
            action = self.display_dock_actions.get(display)
            if action is None or sip.isdeleted(action):
                self.display_dock_actions.pop(display, None)
                continue
        self.menubar.populate_view_menu(
            [dock for dock in self.dock_by_key.values()],
            list(self.display_dock_actions.values()),
        )

    def _display_title(self, display: BaseDisplay) -> str:
        name = getattr(display, "user_label", None)
        if name:
            return name
        try:
            cls = display_registry.get_class(display.type_key)
            return getattr(cls, "DISPLAY_NAME", display.type_key)
        except Exception:
            return display.type_key

    def _on_display_added(self, state: object) -> None:
        if not isinstance(state, BaseDisplay):
            return
        if state in self.display_docks:
            self._on_display_changed(state)
            return

        dock = qtads.CDockWidget(self._display_title(state))
        dock.setObjectName(self._display_dock_object_name(state))
        dock.setWidget(state)
        self.display_docks[state] = dock

        try:
            self.dock_manager.addDockWidget(qtads.DockWidgetArea.RightDockWidgetArea, dock)
        except Exception:
            try:
                self.dock_manager.addDockWidget(qtads.DockWidgetArea.LeftDockWidgetArea, dock)
            except Exception:
                dock.deleteLater()
                self.display_docks.pop(state, None)
                return

        action = QAction(self._display_title(state), self)
        action.setCheckable(True)
        visible = bool(getattr(state, "docked_visible", True))
        action.setChecked(visible)
        action.toggled.connect(lambda checked, display=state: self._on_display_dock_toggled(display, checked))
        self.display_dock_actions[state] = action

        if not visible:
            self._set_display_visibility(state, False, update_action=False)
        else:
            self._set_display_visibility(state, True, update_action=False)

        if hasattr(dock, "closed"):
            dock.closed.connect(lambda *_args, display=state: self._on_display_dock_closed(display))

        self._refresh_view_menu()

    def _on_display_removed(self, state: object) -> None:
        if not isinstance(state, BaseDisplay):
            return
        dock = self.display_docks.pop(state, None)
        action = self.display_dock_actions.pop(state, None)

        if action is not None:
            try:
                action.toggled.disconnect()
            except Exception:
                pass
            try:
                self.menubar.view_menu.removeAction(action)
            except Exception:
                pass
            action.setParent(None)
            action.deleteLater()

        if dock is not None:
            if sip.isdeleted(dock):
                dock = None

            if dock is not None:
                try:
                    if hasattr(self.dock_manager, "removeDockWidget"):
                        self.dock_manager.removeDockWidget(dock)
                except Exception:
                    pass
                dock.setWidget(None)
                dock.deleteLater()

        self._refresh_view_menu()

    def _display_dock_object_name(self, display: BaseDisplay) -> str:
        instance_id = str(getattr(display, "instance_id", "")).strip()
        if not instance_id:
            instance_id = str(id(display))
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in instance_id)
        return f"dock.display.{safe}"

    def _on_display_changed(self, state: object) -> None:
        if not isinstance(state, BaseDisplay):
            return
        dock = self.display_docks.get(state)
        action = self.display_dock_actions.get(state)
        if dock is not None and sip.isdeleted(dock):
            self.display_docks.pop(state, None)
            dock = None
        if action is not None and sip.isdeleted(action):
            self.display_dock_actions.pop(state, None)
            action = None

        name = self._display_title(state)
        if dock is not None:
            try:
                dock.setWindowTitle(name)
            except Exception:
                pass
        if action is not None:
            action.setText(name)
            if action.isChecked() != bool(getattr(state, "docked_visible", True)):
                action.blockSignals(True)
                action.setChecked(bool(getattr(state, "docked_visible", True)))
                action.blockSignals(False)

        if dock is not None:
            desired_visible = bool(getattr(state, "docked_visible", True))
            current_visible = bool(dock.toggleViewAction().isChecked())
            if current_visible != desired_visible:
                self._set_display_visibility(state, desired_visible, update_action=False)
        self._refresh_view_menu()

    def _display_active(self, display: BaseDisplay) -> bool:
        return display in self.display_service.app_state.displays

    def _on_display_dock_toggled(self, display: BaseDisplay, visible: bool) -> None:
        if not self._display_active(display):
            return
        if bool(getattr(display, "docked_visible", True)) == bool(visible):
            return
        self.display_service.set_dock_visibility(display, visible)

    def _on_display_dock_closed(self, display: BaseDisplay) -> None:
        if not self._display_active(display):
            return
        try:
            self.display_service.detach(display)
        except Exception:
            pass
        try:
            self.display_service.set_dock_visibility(display, False)
        except Exception:
            self._set_display_visibility(display, False)

    def _set_display_visibility(self, display: BaseDisplay, visible: bool, update_action: bool = True) -> None:
        dock = self.display_docks.get(display)
        if dock is None or sip.isdeleted(dock):
            if dock is not None:
                self.display_docks.pop(display, None)
            return

        if dock is not None:
            dock.toggleView(visible)
        action = self.display_dock_actions.get(display)
        if action is not None and sip.isdeleted(action):
            self.display_dock_actions.pop(display, None)
            action = None
        if action is not None and update_action:
            if action.isChecked() != visible:
                action.blockSignals(True)
                action.setChecked(visible)
                action.blockSignals(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closing.emit()
        super().closeEvent(event)
