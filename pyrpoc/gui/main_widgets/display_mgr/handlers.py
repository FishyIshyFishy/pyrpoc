from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem, QMessageBox

from pyrpoc.displays.display_registry import display_registry
from pyrpoc.gui.main_widgets.display_mgr.forms import prompt_display_parameters

if TYPE_CHECKING:
    from pyrpoc.gui.main_widgets.display_mgr.widget import DisplayManagerWidget


def refresh_available(widget: DisplayManagerWidget) -> None:
    """
    Build display dropdown constrained by selected modality contract.

    Route:
    - modality selected signal
    - -> this handler
    - -> display list filtered by output type and allowed display keys.
    """
    contract = widget.modality_service.get_selected_contract()
    output_contract = contract.get("output_data_contract")
    allowed_displays = set(contract.get("allowed_displays", []))
    current_key = widget._selected_display_key()
    available_rows = {row["key"]: row for row in widget.display_service.list_available()}

    widget.display_combo.clear()
    if isinstance(output_contract, str) and output_contract.strip():
        keys = widget.display_service.list_compatible_with(output_contract.strip())
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
    """
    Rebuild visible instance list from service inventory rows.

    Route:
    - display add/remove/attach/detach signal
    - -> this handler
    - -> list widget rows with stable object identity in UserRole.
    """
    current = widget._selected_display()
    current_id = id(current) if current is not None else None
    widget.instances_list.blockSignals(True)
    try:
        widget.instances_list.clear()
        for idx, row in enumerate(widget.display_service.list_instances(), start=1):
            raw_display_id = row.get("display_id")
            display_id: int | None = None
            if isinstance(raw_display_id, int):
                display_id = raw_display_id
            elif isinstance(raw_display_id, bool):
                display_id = int(raw_display_id)
            else:
                state_value = row.get("state")
                if isinstance(state_value, int):
                    display_id = state_value
                elif state_value is not None:
                    try:
                        display_id = id(state_value)
                    except Exception:
                        display_id = None

            if display_id is None:
                continue
            marker = "attached" if row["attached"] else "detached"
            item = QListWidgetItem(f"{row['name']} [{idx}] ({marker})")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setData(Qt.ItemDataRole.UserRole, display_id)
            widget.instances_list.addItem(item)

        if current_id is not None:
            for idx in range(widget.instances_list.count()):
                item = widget.instances_list.item(idx)
                if item.data(Qt.ItemDataRole.UserRole) == current_id:
                    widget.instances_list.setCurrentRow(idx)
                    break
    finally:
        widget.instances_list.blockSignals(False)


def on_display_name_edited(widget: DisplayManagerWidget, item: Any) -> None:
    if not isinstance(item, QListWidgetItem):
        return
    raw = item.text().strip()
    if not raw:
        return
    display_id = item.data(Qt.ItemDataRole.UserRole)
    if not isinstance(display_id, int):
        return
    display = widget.display_service.get_display_by_id(display_id)
    if display is None:
        return

    name = raw.split(" [", 1)[0].strip()
    if not name:
        return
    widget.display_service.set_display_name(display, name)


def on_add_clicked(widget: DisplayManagerWidget) -> None:
    """
    Add one display using the selected registry key and optional parameter dialog.

    Route:
    - Add button click
    - -> this handler
    - -> `DisplayService.create_display`
    - -> `display_added` tab/list refresh.
    """
    key = widget._selected_display_key()
    if not key:
        show_error(widget, "No compatible display available for selected modality")
        return

    raw_settings: dict[str, Any] = {}
    display_cls = display_registry.get_class(key)
    display_name = widget.name_input.text().strip()
    if not display_name:
        row = widget.display_service.list_available()
        fallback = next((row_entry for row_entry in row if row_entry["key"] == key), None)
        display_name = str(fallback.get("display_name", key)) if isinstance(fallback, dict) else key

    if any(display_cls.DISPLAY_PARAMETERS.values()):
        settings = prompt_display_parameters(widget, display_cls.DISPLAY_PARAMETERS)
        if settings is None:
            return
        raw_settings = settings

    try:
        widget.display_service.create_display(key, raw_settings, user_label=display_name)
        widget.status_label.setText("Status: added display")
        widget.name_input.clear()
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


def on_display_changed(widget: DisplayManagerWidget, state: object) -> None:
    del state
    widget.status_label.setText("Status: display updated")
    refresh_instances(widget)


def on_display_removed(widget: DisplayManagerWidget, state: object) -> None:
    del state
    refresh_instances(widget)


def on_display_error(widget: DisplayManagerWidget, _state: object, message: str) -> None:
    widget.status_label.setText(f"Status: display error - {message}")


def on_modality_selected(widget: DisplayManagerWidget, key: str) -> None:
    del key
    refresh_available(widget)


def show_error(widget: DisplayManagerWidget, message: str) -> None:
    widget.status_label.setText(f"Status: error - {message}")
    QMessageBox.critical(widget, "Display Error", message)
