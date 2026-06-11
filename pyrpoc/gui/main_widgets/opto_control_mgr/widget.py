from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from . import handlers
from pyrpoc.gui.main_widgets.opto_control_mgr.state import OptoControlManagerState
from pyrpoc.gui.main_widgets.opto_control_mgr.ui import build_opto_control_manager_ui
from pyrpoc.services.display_service import DisplayService
from pyrpoc.services.modality_service import ModalityService
from pyrpoc.services.opto_control_service import OptoControlService


class OptoControlManagerWidget(QWidget):
    def __init__(
        self,
        opto_control_service: OptoControlService,
        modality_service: ModalityService,
        display_service: DisplayService,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.opto_control_service = opto_control_service
        self.modality_service = modality_service
        self.display_service = display_service

        self.state = OptoControlManagerState()
        self.ui = build_opto_control_manager_ui(self)

        self.type_combo = self.ui.type_combo
        self.add_btn = self.ui.add_btn
        self.instances_layout = self.ui.instances_layout

        self.wire_signals()
        self.refresh_available()
        self.refresh_instances()

    def wire_signals(self) -> None:
        self.add_btn.clicked.connect(self.on_add_clicked)
        self.opto_control_service.inventory_changed.connect(self.refresh_instances)
        self.modality_service.modality_selected.connect(self.on_modality_selected)

    def selected_type_key(self) -> str | None:
        value = self.type_combo.currentData()
        return str(value) if value else None

    def refresh_available(self) -> None:
        handlers.refresh_available(self)

    def refresh_instances(self) -> None:
        '''
        Service inventory changes should only diff cards.
        This preserves expanded card bodies and avoids stale QObject deref on add/remove.
        '''
        handlers.refresh_instances(self)

    def on_add_clicked(self) -> None:
        handlers.on_add_clicked(self)

    def on_modality_selected(self, key: str) -> None:
        handlers.on_modality_selected(self, key)
