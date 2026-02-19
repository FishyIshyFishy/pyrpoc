from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem, QWidget

from pyrpoc.domain.app_state import InstrumentState
from pyrpoc.gui.main_widgets.instrument_mgr.forms import clear_dynamic_panels
from pyrpoc.gui.main_widgets.instrument_mgr.handlers import (
    on_add_clicked,
    on_connect_clicked,
    on_connection_changed,
    on_disconnect_clicked,
    on_instance_selected,
    on_remove_clicked,
    refresh_available,
    refresh_instances,
    run_action,
    show_error,
)
from pyrpoc.gui.main_widgets.instrument_mgr.state import InstrumentManagerState
from pyrpoc.gui.main_widgets.instrument_mgr.ui import build_instrument_manager_ui
from pyrpoc.services.instrument_service import InstrumentService


class InstrumentManagerWidget(QWidget):
    def __init__(self, instrument_service: InstrumentService, parent: QWidget | None = None):
        super().__init__(parent)
        self.instrument_service = instrument_service
        self.state = InstrumentManagerState()
        self.ui = build_instrument_manager_ui(self)

        # Compatibility aliases for code expecting prior attribute names.
        self.type_combo = self.ui.type_combo
        self.add_btn = self.ui.add_btn
        self.instances_list = self.ui.instances_list
        self.connect_btn = self.ui.connect_btn
        self.disconnect_btn = self.ui.disconnect_btn
        self.remove_btn = self.ui.remove_btn
        self.status_label = self.ui.status_label
        self.config_box = self.ui.config_box
        self.config_form = self.ui.config_form
        self.actions_box = self.ui.actions_box
        self.actions_layout = self.ui.actions_layout
        self._config_widgets = self.state.config_widgets
        self._action_widgets = self.state.action_widgets
        self._actions_by_label = self.state.actions_by_label

        self._wire_signals()
        self._refresh_available()
        self._refresh_instances()

    def _wire_signals(self) -> None:
        self.add_btn.clicked.connect(self._on_add_clicked)
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        self.disconnect_btn.clicked.connect(self._on_disconnect_clicked)
        self.remove_btn.clicked.connect(self._on_remove_clicked)
        self.instances_list.currentItemChanged.connect(self._on_instance_selected)

        self.instrument_service.inventory_changed.connect(self._refresh_instances)
        self.instrument_service.connection_changed.connect(self._on_connection_changed)

    def _refresh_available(self) -> None:
        refresh_available(self)

    def _refresh_instances(self) -> None:
        refresh_instances(self)

    def _clear_dynamic_panels(self) -> None:
        clear_dynamic_panels(self.ui, self.state)

    def _on_add_clicked(self) -> None:
        on_add_clicked(self)

    def _on_connect_clicked(self) -> None:
        on_connect_clicked(self)

    def _on_disconnect_clicked(self) -> None:
        on_disconnect_clicked(self)

    def _on_remove_clicked(self) -> None:
        on_remove_clicked(self)

    def _run_action(self, action_label: str) -> None:
        run_action(self, action_label)

    def _on_instance_selected(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        on_instance_selected(self, current, previous)

    def _on_connection_changed(self, state: object, connected: bool) -> None:
        on_connection_changed(self, state, connected)

    def _selected_type_key(self) -> str:
        data = self.type_combo.currentData()
        if isinstance(data, str):
            return data
        return self.type_combo.currentText().strip()

    def _selected_instance(self) -> InstrumentState | None:
        item = self.instances_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(value, InstrumentState):
            return value
        return None

    def _show_error(self, message: str) -> None:
        show_error(self, message)
