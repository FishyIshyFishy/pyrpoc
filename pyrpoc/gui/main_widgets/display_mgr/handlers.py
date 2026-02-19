from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem, QMessageBox

from pyrpoc.backend_utils.data import BaseData
from pyrpoc.domain.app_state import DisplayState
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
    current = widget._selected_display()
    widget.instances_list.clear()
    for idx, row in enumerate(widget.display_service.list_instances(), start=1):
        state: DisplayState = row["state"]
        marker = "attached" if row["attached"] else "detached"
        item = QListWidgetItem(f"{row['name']} [{idx}] ({marker})")
        item.setData(Qt.ItemDataRole.UserRole, state)
        widget.instances_list.addItem(item)

    if current is not None:
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
        widget.display_service.create_display(key, raw_settings)
        widget.status_label.setText("Status: added display")
    except Exception as exc:
        show_error(widget, str(exc))


def on_attach_clicked(widget: DisplayManagerWidget) -> None:
    state = widget._selected_display()
    if state is None:
        return
    widget.display_service.attach(state)
    widget.status_label.setText("Status: attached display")
    refresh_instances(widget)


def on_detach_clicked(widget: DisplayManagerWidget) -> None:
    state = widget._selected_display()
    if state is None:
        return
    widget.display_service.detach(state)
    widget.status_label.setText("Status: detached display")
    refresh_instances(widget)


def on_remove_clicked(widget: DisplayManagerWidget) -> None:
    state = widget._selected_display()
    if state is None:
        return
    widget.display_service.remove_display(state)
    widget.status_label.setText("Status: removed display")


def row_tab_title(state: DisplayState, widget: DisplayManagerWidget) -> str:
    cls = display_registry.get_class(state.type_key)
    return f"{getattr(cls, 'DISPLAY_NAME', state.type_key)} [{id(state)}]"


def on_display_added(widget: DisplayManagerWidget, state: object) -> None:
    if isinstance(state, DisplayState):
        marker = str(id(state))
        exists = False
        for idx in range(widget.display_tabs.count()):
            if widget.display_tabs.tabToolTip(idx) == marker:
                exists = True
                break
        if not exists:
            widget.display_tabs.addTab(state.instance, row_tab_title(state, widget))
            widget.display_tabs.setTabToolTip(widget.display_tabs.count() - 1, marker)
    widget.status_label.setText("Status: display added")
    refresh_instances(widget)


def on_display_removed(widget: DisplayManagerWidget, state: object) -> None:
    if not isinstance(state, DisplayState):
        refresh_instances(widget)
        return
    for idx in range(widget.display_tabs.count()):
        if widget.display_tabs.tabToolTip(idx) == str(id(state)):
            tab = widget.display_tabs.widget(idx)
            widget.display_tabs.removeTab(idx)
            del tab
            break
    refresh_instances(widget)


def on_display_error(widget: DisplayManagerWidget, _state: object, message: str) -> None:
    widget.status_label.setText(f"Status: display error - {message}")


def on_modality_selected(widget: DisplayManagerWidget, key: str) -> None:
    del key
    refresh_available(widget)


def show_error(widget: DisplayManagerWidget, message: str) -> None:
    widget.status_label.setText(f"Status: error - {message}")
    QMessageBox.critical(widget, "Display Error", message)
