from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from pyrpoc.gui.main_widgets.display_mgr import handlers
from pyrpoc.gui.main_widgets.display_mgr.state import DisplayManagerState
from pyrpoc.gui.main_widgets.display_mgr.ui import build_display_manager_ui
from pyrpoc.services.display_service import DisplayService
from pyrpoc.services.modality_service import ModalityService


class DisplayManagerWidget(QWidget):
    def __init__(
        self,
        display_service: DisplayService,
        modality_service: ModalityService,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.display_service = display_service
        self.modality_service = modality_service
        self.state = DisplayManagerState()
        self.ui = build_display_manager_ui(self)

        self.display_combo = self.ui.display_combo
        self.add_btn = self.ui.add_btn
        self.instances_layout = self.ui.instances_layout

        self.wire_signals()
        self.refresh_available()
        self.refresh_instances()

    def wire_signals(self) -> None:
        self.add_btn.clicked.connect(self.on_add_clicked)

        self.display_service.display_added.connect(lambda _s: self.refresh_instances())
        self.display_service.display_removed.connect(lambda _s: self.refresh_instances())
        self.display_service.display_changed.connect(lambda _s: self.refresh_instances())
        self.display_service.display_error.connect(self.on_display_error)

        self.modality_service.modality_selected.connect(self.on_modality_selected)

    def refresh_available(self) -> None:
        handlers.refresh_available(self)

    def refresh_instances(self) -> None:
        handlers.refresh_instances(self)

    def on_add_clicked(self) -> None:
        handlers.on_add_clicked(self)

    def on_display_error(self, state: object, message: str) -> None:
        handlers.on_display_error(self, state, message)

    def on_modality_selected(self, key: str) -> None:
        handlers.on_modality_selected(self, key)

    def selected_display_key(self) -> str:
        data = self.display_combo.currentData()
        if isinstance(data, str):
            return data
        return self.display_combo.currentText().strip()
