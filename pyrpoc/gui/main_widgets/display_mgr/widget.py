from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QAbstractItemView, QWidget

from pyrpoc.displays.base_display import BaseDisplay
from pyrpoc.gui.main_widgets.display_mgr.handlers import (
    on_add_clicked,
    on_attach_clicked,
    on_detach_clicked,
    on_display_name_edited,
    on_display_error,
    on_modality_selected,
    on_remove_clicked,
    refresh_available,
    refresh_instances,
    show_error,
)
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

        # Compatibility aliases for existing callers.
        self.display_combo = self.ui.display_combo
        self.add_btn = self.ui.add_btn
        self.instances_list = self.ui.instances_list
        self.instances_list.setEditTriggers(
            QAbstractItemView.EditTrigger.AllEditTriggers
        )
        self.instances_list.itemDoubleClicked.connect(self._on_display_name_edit_request)
        self.attach_btn = self.ui.attach_btn
        self.detach_btn = self.ui.detach_btn
        self.remove_btn = self.ui.remove_btn
        self.status_label = self.ui.status_label
        self.name_input = self.ui.name_input

        self._wire_signals()
        self._refresh_available()
        self._refresh_instances()

    def _wire_signals(self) -> None:
        self.add_btn.clicked.connect(self._on_add_clicked)
        self.attach_btn.clicked.connect(self._on_attach_clicked)
        self.detach_btn.clicked.connect(self._on_detach_clicked)
        self.remove_btn.clicked.connect(self._on_remove_clicked)

        self.display_service.display_added.connect(lambda _state: self._refresh_instances())
        self.display_service.display_removed.connect(lambda _state: self._refresh_instances())
        self.display_service.display_changed.connect(lambda _state: self._refresh_instances())
        self.display_service.display_error.connect(self._on_display_error)
        self.instances_list.itemChanged.connect(self._on_display_name_edited)
        self.modality_service.modality_selected.connect(self._on_modality_selected)

    def _refresh_available(self) -> None:
        refresh_available(self)

    def _refresh_instances(self) -> None:
        refresh_instances(self)

    def _on_add_clicked(self) -> None:
        on_add_clicked(self)

    def _on_attach_clicked(self) -> None:
        on_attach_clicked(self)

    def _on_detach_clicked(self) -> None:
        on_detach_clicked(self)

    def _on_remove_clicked(self) -> None:
        on_remove_clicked(self)

    def _on_display_error(self, state: object, message: str) -> None:
        on_display_error(self, state, message)

    def _on_display_name_edit_request(self, item) -> None:
        if item is None:
            return
        self.instances_list.editItem(item)

    def _on_modality_selected(self, key: str) -> None:
        on_modality_selected(self, key)

    def _selected_display(self) -> BaseDisplay | None:
        item = self.instances_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(value, int):
            return None
        display = self.display_service.get_display_by_id(value)
        return display

    def _selected_display_key(self) -> str:
        data = self.display_combo.currentData()
        if isinstance(data, str):
            return data
        return self.display_combo.currentText().strip()

    def _show_error(self, message: str) -> None:
        show_error(self, message)

    def _on_display_name_edited(self, item) -> None:
        on_display_name_edited(self, item)
