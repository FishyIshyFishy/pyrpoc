from __future__ import annotations

from PyQt6.QtWidgets import QVBoxLayout, QWidget
import PyQt6Ads as qtads

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
        self.menubar = MainMenuBar(self)

        dock_acq = self.add_dock(
            "Acquisition Manager",
            AcquisitionManagerWidget(self.modality_service),
            qtads.DockWidgetArea.LeftDockWidgetArea,
        )
        self.add_dock(
            "Instrument Manager",
            InstrumentManagerWidget(self.instrument_service),
            qtads.DockWidgetArea.LeftDockWidgetArea,
            tab_with=dock_acq,
        )
        self.add_dock(
            "Display Manager",
            DisplayManagerWidget(self.display_service, self.modality_service),
            qtads.DockWidgetArea.LeftDockWidgetArea,
            tab_with=dock_acq,
        )
        self.add_dock(
            "Opto-Control Manager",
            OptoControlManagerWidget(
                self.opto_control_service,
                self.modality_service,
                self.display_service,
            ),
            qtads.DockWidgetArea.LeftDockWidgetArea,
            tab_with=dock_acq,
        )

        layout = QVBoxLayout(self)
        layout.setMenuBar(self.menubar)
        layout.addWidget(self.dock_manager)

        self.menubar.populate_view_menu(self.docks)
        selected_mode = self.theme_controller.get_saved_mode()
        self.menubar.populate_style_menu(selected_mode)
        self.menubar.style_selected.connect(self.set_style)

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
