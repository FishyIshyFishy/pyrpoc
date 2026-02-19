from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem, QMessageBox

from pyrpoc.domain.app_state import InstrumentState
from pyrpoc.gui.main_widgets.instrument_mgr.forms import (
    build_actions_area,
    build_config_form,
    clear_dynamic_panels,
    collect_values,
)
from pyrpoc.instruments.instrument_registry import instrument_registry

if TYPE_CHECKING:
    from pyrpoc.gui.main_widgets.instrument_mgr.widget import InstrumentManagerWidget


def refresh_available(widget: InstrumentManagerWidget) -> None:
    current_key = widget._selected_type_key()
    widget.type_combo.blockSignals(True)
    widget.type_combo.clear()
    for row in widget.instrument_service.list_available():
        key = row["key"]
        name = row.get("display_name", key)
        widget.type_combo.addItem(name, key)
    widget.type_combo.blockSignals(False)

    if current_key:
        idx = widget.type_combo.findData(current_key)
        if idx >= 0:
            widget.type_combo.setCurrentIndex(idx)
    elif widget.type_combo.count() > 0:
        widget.type_combo.setCurrentIndex(0)


def refresh_instances(widget: InstrumentManagerWidget) -> None:
    previous = widget._selected_instance()
    selected_found = False
    widget.instances_list.blockSignals(True)
    widget.instances_list.clear()
    for idx, row in enumerate(widget.instrument_service.list_instances(), start=1):
        state: InstrumentState = row["state"]
        marker = "connected" if row["connected"] else "disconnected"
        item = QListWidgetItem(f"{row['name']} [{idx}] ({marker})")
        item.setData(Qt.ItemDataRole.UserRole, state)
        widget.instances_list.addItem(item)
    widget.instances_list.blockSignals(False)

    if previous is not None:
        for idx in range(widget.instances_list.count()):
            item = widget.instances_list.item(idx)
            if item.data(Qt.ItemDataRole.UserRole) == previous:
                widget.instances_list.setCurrentRow(idx)
                selected_found = True
                break
    if not selected_found:
        clear_dynamic_panels(widget.ui, widget.state)


def on_add_clicked(widget: InstrumentManagerWidget) -> None:
    key = widget._selected_type_key()
    if not key:
        return

    try:
        widget.instrument_service.create_instrument(key)
        widget.status_label.setText("Status: added instrument")
        refresh_instances(widget)
    except Exception as exc:
        show_error(widget, str(exc))


def on_connect_clicked(widget: InstrumentManagerWidget) -> None:
    state = widget._selected_instance()
    if state is None:
        show_error(widget, "Select an instrument instance first")
        return
    try:
        config = collect_values(widget.state.config_widgets)
        widget.instrument_service.connect(state, config)
    except Exception as exc:
        show_error(widget, str(exc))


def on_disconnect_clicked(widget: InstrumentManagerWidget) -> None:
    state = widget._selected_instance()
    if state is None:
        show_error(widget, "Select an instrument instance first")
        return
    try:
        widget.instrument_service.disconnect(state)
    except Exception as exc:
        show_error(widget, str(exc))


def on_remove_clicked(widget: InstrumentManagerWidget) -> None:
    state = widget._selected_instance()
    if state is None:
        return
    widget.instrument_service.remove_instrument(state)
    widget.status_label.setText("Status: removed instrument")


def run_action(widget: InstrumentManagerWidget, action_label: str) -> None:
    state = widget._selected_instance()
    if state is None:
        show_error(widget, "Select an instrument instance first")
        return

    action = widget.state.actions_by_label.get(action_label)
    if action is None:
        show_error(widget, f"Unknown action '{action_label}'")
        return

    if action.dangerous:
        prompt = action.confirm_text or f"Run action '{action.label}'?"
        response = QMessageBox.question(
            widget,
            "Confirm Action",
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return

    raw_args = collect_values(widget.state.action_widgets.get(action_label, {}))
    try:
        widget.instrument_service.run_action(state, action_label, raw_args)
        widget.status_label.setText(f"Status: ran '{action_label}'")
    except Exception as exc:
        show_error(widget, str(exc))


def on_instance_selected(
    widget: InstrumentManagerWidget,
    current: QListWidgetItem | None,
    previous: QListWidgetItem | None,
) -> None:
    del previous
    if current is None:
        clear_dynamic_panels(widget.ui, widget.state)
        return

    state = widget._selected_instance()
    if state is None:
        clear_dynamic_panels(widget.ui, widget.state)
        return

    try:
        key = widget.instrument_service.get_instance_key(state)
        cls = instrument_registry.get_class(key)
        build_config_form(widget.ui, widget.state, cls.CONFIG_PARAMETERS)
        build_actions_area(
            widget.ui,
            widget.state,
            cls.ACTIONS,
            lambda action_name: run_action(widget, action_name),
        )
    except Exception:
        clear_dynamic_panels(widget.ui, widget.state)


def on_connection_changed(widget: InstrumentManagerWidget, _state: object, connected: bool) -> None:
    state = "connected" if connected else "disconnected"
    widget.status_label.setText(f"Status: instrument {state}")
    refresh_instances(widget)


def show_error(widget: InstrumentManagerWidget, message: str) -> None:
    widget.status_label.setText(f"Status: error - {message}")
    QMessageBox.critical(widget, "Instrument Error", message)
