from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem, QMessageBox

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
    previous = widget._selected_instance_id()
    selected_found = False
    widget.instances_list.blockSignals(True)
    widget.instances_list.clear()
    for row in widget.instrument_service.list_instances():
        marker = "connected" if row["connected"] else "disconnected"
        item = QListWidgetItem(f"{row['name']} [{row['instance_id']}] ({marker})")
        item.setData(Qt.ItemDataRole.UserRole, row["instance_id"])
        widget.instances_list.addItem(item)
    widget.instances_list.blockSignals(False)

    if previous:
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
        instance_id, _ = widget.instrument_service.create_instrument(key)
        widget.status_label.setText(f"Status: added {instance_id}")
        refresh_instances(widget)
    except Exception as exc:
        show_error(widget, str(exc))


def on_connect_clicked(widget: InstrumentManagerWidget) -> None:
    instance_id = widget._selected_instance_id()
    if not instance_id:
        show_error(widget, "Select an instrument instance first")
        return
    try:
        config = collect_values(widget.state.config_widgets)
        widget.instrument_service.connect(instance_id, config)
    except Exception as exc:
        show_error(widget, str(exc))


def on_disconnect_clicked(widget: InstrumentManagerWidget) -> None:
    instance_id = widget._selected_instance_id()
    if not instance_id:
        show_error(widget, "Select an instrument instance first")
        return
    try:
        widget.instrument_service.disconnect(instance_id)
    except Exception as exc:
        show_error(widget, str(exc))


def on_remove_clicked(widget: InstrumentManagerWidget) -> None:
    instance_id = widget._selected_instance_id()
    if not instance_id:
        return
    widget.instrument_service.remove_instrument(instance_id)
    widget.status_label.setText(f"Status: removed {instance_id}")


def run_action(widget: InstrumentManagerWidget, action_label: str) -> None:
    instance_id = widget._selected_instance_id()
    if not instance_id:
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
        widget.instrument_service.run_action(instance_id, action_label, raw_args)
        widget.status_label.setText(f"Status: ran '{action_label}' on {instance_id}")
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

    instance_id = widget._selected_instance_id()
    if not instance_id:
        clear_dynamic_panels(widget.ui, widget.state)
        return

    try:
        key = widget.instrument_service.get_instance_key(instance_id)
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


def on_connection_changed(widget: InstrumentManagerWidget, instance_id: str, connected: bool) -> None:
    state = "connected" if connected else "disconnected"
    widget.status_label.setText(f"Status: {instance_id} {state}")
    refresh_instances(widget)


def show_error(widget: InstrumentManagerWidget, message: str) -> None:
    widget.status_label.setText(f"Status: error - {message}")
    QMessageBox.critical(widget, "Instrument Error", message)
