from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PyQt6.QtCore import QByteArray
from PyQt6.QtWidgets import QVBoxLayout, QWidget
import PyQt6Ads as qtads

from pyrpoc.domain.session_state import GuiLayoutSessionState
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
        self.menubar = MainMenuBar(self)
        self._build_default_docks()

        layout = QVBoxLayout(self)
        layout.setMenuBar(self.menubar)
        layout.addWidget(self.dock_manager)

        self.menubar.populate_view_menu(self.docks)
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
        self.dock_by_key[dock_specs[0].key] = dock_acq

        dock_instruments = self.add_dock(
            dock_specs[1].title,
            InstrumentManagerWidget(self.instrument_service),
            dock_specs[1].area,
            tab_with=dock_acq,
        )
        self.dock_by_key[dock_specs[1].key] = dock_instruments

        dock_displays = self.add_dock(
            dock_specs[2].title,
            DisplayManagerWidget(self.display_service, self.modality_service),
            dock_specs[2].area,
            tab_with=dock_acq,
        )
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

    def capture_layout_state(self) -> GuiLayoutSessionState:
        raw = self.dock_manager.saveState()
        encoded = bytes(raw.toBase64()).decode("ascii") if isinstance(raw, QByteArray) else None
        dock_visibility = {
            key.value: bool(dock.toggleViewAction().isChecked()) for key, dock in self.dock_by_key.items()
        }
        return GuiLayoutSessionState(ads_state_base64=encoded, dock_visibility=dock_visibility)

    def restore_layout_state(self, state: GuiLayoutSessionState) -> None:
        restored = False
        if state.ads_state_base64:
            raw = QByteArray.fromBase64(QByteArray(state.ads_state_base64.encode("ascii")))
            restored = bool(self.dock_manager.restoreState(raw))
        if not restored:
            self.restore_default_layout()
            for key, visible in state.dock_visibility.items():
                for dock_key, dock in self.dock_by_key.items():
                    if dock_key.value == key:
                        dock.toggleView(visible)
                        break

    def restore_default_layout(self) -> None:
        for key in (DockKey.ACQUISITION, DockKey.INSTRUMENTS, DockKey.DISPLAYS, DockKey.OPTOCONTROLS):
            dock = self.dock_by_key.get(key)
            if dock is not None:
                dock.toggleView(True)
