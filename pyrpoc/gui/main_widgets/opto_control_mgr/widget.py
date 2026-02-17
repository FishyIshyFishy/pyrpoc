from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from pyrpoc.gui.main_widgets.opto_control_mgr.handlers import (
    init_editor_host,
    on_add_clicked,
    on_card_enable_toggled,
    on_card_expand_requested,
    on_card_remove_requested,
    on_modality_selected,
    refresh_available,
    refresh_instances,
    run_action,
    show_error,
    sync_controls_from_status,
)
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

        # Compatibility aliases for existing callers.
        self.type_combo = self.ui.type_combo
        self.add_btn = self.ui.add_btn
        self.status_label = self.ui.status_label
        self.instances_layout = self.ui.instances_layout
        self.editor_host_layout = self.ui.editor_host_layout

        self._wire_signals()
        self._init_editor_host()
        self._refresh_available()
        self._refresh_instances()

    def _wire_signals(self) -> None:
        self.add_btn.clicked.connect(self._on_add_clicked)

        self.opto_control_service.inventory_changed.connect(self._refresh_instances)
        self.modality_service.modality_selected.connect(self._on_modality_selected)

    def _refresh_available(self) -> None:
        refresh_available(self)

    def _init_editor_host(self) -> None:
        init_editor_host(self)

    def _refresh_instances(self) -> None:
        refresh_instances(self)

    def _on_add_clicked(self) -> None:
        on_add_clicked(self)

    def _run_action(self, action_label: str) -> None:
        instance_id = self._expanded_instance_id()
        if instance_id:
            run_action(self, instance_id, action_label)

    def _on_card_expand_requested(self, instance_id: str) -> None:
        on_card_expand_requested(self, instance_id)

    def _on_card_enable_toggled(self, instance_id: str, checked: bool) -> None:
        on_card_enable_toggled(self, instance_id, checked)

    def _on_card_remove_requested(self, instance_id: str) -> None:
        on_card_remove_requested(self, instance_id)

    def _sync_controls_from_status(self, instance_id: str) -> None:
        sync_controls_from_status(self, instance_id)

    def _on_modality_selected(self, key: str) -> None:
        on_modality_selected(self, key)

    def _selected_type_key(self) -> str:
        data = self.type_combo.currentData()
        if isinstance(data, str):
            return data
        return self.type_combo.currentText().strip()

    def _expanded_instance_id(self) -> str:
        return self.state.expanded_instance_id or ""

    def _set_expanded_instance_id(self, instance_id: str | None) -> None:
        self.state.expanded_instance_id = instance_id

    def _show_error(self, message: str) -> None:
        show_error(self, message)

    def _card_for(self, instance_id: str):
        return self.state.card_widgets.get(instance_id)
