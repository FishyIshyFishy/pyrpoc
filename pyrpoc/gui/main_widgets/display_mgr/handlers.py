from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem, QMessageBox

from pyrpoc.backend_utils.data import BaseData
from pyrpoc.displays.display_registry import display_registry
from pyrpoc.gui.main_widgets.display_mgr.forms import prompt_display_parameters

if TYPE_CHECKING:
    from pyrpoc.gui.main_widgets.display_mgr.widget import DisplayManagerWidget


def refresh_available(widget: DisplayManagerWidget) -> None:
    contract = widget.modality_service.get_selected_contract()
    output_type = contract.get("output_data_type")
    allowed_displays = set(contract.get("allowed_displays", []))
    current_key = widget._selected_display_key()
    available_rows = {row["key"]: row for row in widget.display_service.list_available()}

    widget.display_combo.clear()
    if isinstance(output_type, type) and issubclass(output_type, BaseData):
        keys = widget.display_service.list_compatible_with(output_type)
    else:
        keys = list(available_rows.keys())

    if allowed_displays:
        keys = [key for key in keys if key in allowed_displays]

    for key in keys:
        row = available_rows.get(key, {"display_name": key})
        widget.display_combo.addItem(row.get("display_name", key), key)

    if current_key:
        idx = widget.display_combo.findData(current_key)
        if idx >= 0:
            widget.display_combo.setCurrentIndex(idx)


def refresh_instances(widget: DisplayManagerWidget) -> None:
    current = widget._selected_display_id()
    widget.instances_list.clear()
    for row in widget.display_service.list_instances():
        marker = "attached" if row["attached"] else "detached"
        item = QListWidgetItem(f"{row['name']} [{row['display_id']}] ({marker})")
        item.setData(Qt.ItemDataRole.UserRole, row["display_id"])
        widget.instances_list.addItem(item)

    if current:
        for idx in range(widget.instances_list.count()):
            item = widget.instances_list.item(idx)
            if item.data(Qt.ItemDataRole.UserRole) == current:
                widget.instances_list.setCurrentRow(idx)
                break


def on_add_clicked(widget: DisplayManagerWidget) -> None:
    key = widget._selected_display_key()
    if not key:
        show_error(widget, "No compatible display available for selected modality")
        return

    raw_settings: dict[str, Any] = {}
    display_cls = display_registry.get_class(key)
    if any(display_cls.DISPLAY_PARAMETERS.values()):
        settings = prompt_display_parameters(widget, display_cls.DISPLAY_PARAMETERS)
        if settings is None:
            return
        raw_settings = settings

    try:
        display_id, display_widget = widget.display_service.create_display(key, raw_settings)
        widget.display_tabs.addTab(display_widget, display_id)
        widget.status_label.setText(f"Status: added {display_id}")
    except Exception as exc:
        show_error(widget, str(exc))


def on_attach_clicked(widget: DisplayManagerWidget) -> None:
    display_id = widget._selected_display_id()
    if not display_id:
        return
    widget.display_service.attach(display_id)
    widget.status_label.setText(f"Status: attached {display_id}")
    refresh_instances(widget)


def on_detach_clicked(widget: DisplayManagerWidget) -> None:
    display_id = widget._selected_display_id()
    if not display_id:
        return
    widget.display_service.detach(display_id)
    widget.status_label.setText(f"Status: detached {display_id}")
    refresh_instances(widget)


def on_remove_clicked(widget: DisplayManagerWidget) -> None:
    display_id = widget._selected_display_id()
    if not display_id:
        return
    widget.display_service.remove_display(display_id)
    widget.status_label.setText(f"Status: removed {display_id}")


def on_display_added(widget: DisplayManagerWidget, display_id: str) -> None:
    widget.status_label.setText(f"Status: display added {display_id}")
    refresh_instances(widget)


def on_display_removed(widget: DisplayManagerWidget, display_id: str) -> None:
    for idx in range(widget.display_tabs.count()):
        if widget.display_tabs.tabText(idx) == display_id:
            tab = widget.display_tabs.widget(idx)
            widget.display_tabs.removeTab(idx)
            if tab is not None:
                tab.deleteLater()
            break
    refresh_instances(widget)


def on_display_error(widget: DisplayManagerWidget, display_id: str, message: str) -> None:
    widget.status_label.setText(f"Status: {display_id} error - {message}")


def on_modality_selected(widget: DisplayManagerWidget, key: str) -> None:
    del key
    refresh_available(widget)


def show_error(widget: DisplayManagerWidget, message: str) -> None:
    widget.status_label.setText(f"Status: error - {message}")
    QMessageBox.critical(widget, "Display Error", message)
