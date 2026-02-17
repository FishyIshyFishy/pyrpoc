from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.backend_utils.contracts import Action, Parameter
from pyrpoc.gui.main_widgets.opto_control_mgr.forms import collect_values, make_editor
from pyrpoc.gui.main_widgets.opto_control_mgr.instance_card import InstanceCardWidget
from pyrpoc.gui.main_widgets.opto_control_mgr.mask_editor import MaskEditorWidget
from pyrpoc.optocontrols.opto_control_registry import opto_control_registry

if TYPE_CHECKING:
    from pyrpoc.gui.main_widgets.opto_control_mgr.widget import OptoControlManagerWidget


def refresh_available(widget: OptoControlManagerWidget) -> None:
    current_key = widget._selected_type_key()
    widget.type_combo.blockSignals(True)
    widget.type_combo.clear()
    for row in widget.opto_control_service.list_available():
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


def init_editor_host(widget: OptoControlManagerWidget) -> None:
    _set_editor_host_placeholder(widget)


def _detach_editor_layout_items(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        child = item.widget()
        if child is not None:
            child.setParent(None)
            child.deleteLater()


def _set_editor_host_placeholder(widget: OptoControlManagerWidget) -> None:
    _detach_editor_layout_items(widget.editor_host_layout)
    placeholder = QLabel("Mask editor area reserved.", widget.ui.editor_host_box)
    widget.editor_host_layout.addWidget(placeholder)
    widget.editor_host_layout.addStretch(1)


def _mount_editor(widget: OptoControlManagerWidget, editor: QWidget, instance_id: str) -> None:
    _detach_editor_layout_items(widget.editor_host_layout)
    widget.editor_host_layout.addWidget(editor)
    widget.state.active_mask_editor_widget = editor
    widget.state.active_mask_editor_instance_id = instance_id


def _unmount_editor(widget: OptoControlManagerWidget) -> None:
    editor = widget.state.active_mask_editor_widget
    widget.state.active_mask_editor_widget = None
    widget.state.active_mask_editor_instance_id = None
    if editor is not None:
        editor.setParent(None)
        editor.deleteLater()
    _set_editor_host_placeholder(widget)


def _cleanup_temp_mask_file(widget: OptoControlManagerWidget, instance_id: str) -> None:
    path = widget.state.mask_tempfile_by_instance.pop(instance_id, None)
    if path is None:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _detach_layout_items(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        child = item.widget()
        if child is not None:
            child.setParent(None)


def _remove_instance_state(widget: OptoControlManagerWidget, instance_id: str) -> None:
    widget.state.config_widgets_by_instance.pop(instance_id, None)
    widget.state.config_params_by_instance.pop(instance_id, None)
    widget.state.action_widgets_by_instance.pop(instance_id, None)
    widget.state.actions_by_label_by_instance.pop(instance_id, None)
    widget.state.method_to_action_by_instance.pop(instance_id, None)
    widget.state.enable_guard_by_instance.pop(instance_id, None)
    widget.state.mask_path_widgets_by_instance.pop(instance_id, None)
    widget.state.mask_create_buttons_by_instance.pop(instance_id, None)
    _cleanup_temp_mask_file(widget, instance_id)


def _clear_instance_form_state(widget: OptoControlManagerWidget, instance_id: str) -> None:
    widget.state.config_widgets_by_instance.pop(instance_id, None)
    widget.state.config_params_by_instance.pop(instance_id, None)
    widget.state.action_widgets_by_instance.pop(instance_id, None)
    widget.state.actions_by_label_by_instance.pop(instance_id, None)
    widget.state.method_to_action_by_instance.pop(instance_id, None)
    widget.state.mask_path_widgets_by_instance.pop(instance_id, None)
    widget.state.mask_create_buttons_by_instance.pop(instance_id, None)


def refresh_instances(widget: OptoControlManagerWidget) -> None:
    rows = widget.opto_control_service.list_instances()
    row_by_id = {row["instance_id"]: row for row in rows}
    current_ids = set(widget.state.card_widgets.keys())
    new_ids = set(row_by_id.keys())

    for removed_id in sorted(current_ids - new_ids):
        if widget.state.active_mask_editor_instance_id == removed_id:
            _close_mask_editor(widget, force_discard=True)
        card = widget.state.card_widgets.pop(removed_id)
        card.setParent(None)
        card.deleteLater()
        _remove_instance_state(widget, removed_id)
        if widget._expanded_instance_id() == removed_id:
            widget._set_expanded_instance_id(None)

    for row in rows:
        instance_id = row["instance_id"]
        card = widget.state.card_widgets.get(instance_id)
        if card is None:
            card = InstanceCardWidget(
                instance_id=instance_id,
                title=f"{row['name']} [{instance_id}]",
                parent=widget.ui.instances_content,
            )
            card.expand_requested.connect(widget._on_card_expand_requested)
            card.enable_toggled.connect(widget._on_card_enable_toggled)
            card.remove_requested.connect(widget._on_card_remove_requested)
            widget.state.card_widgets[instance_id] = card
            widget.state.enable_guard_by_instance[instance_id] = False
        try:
            card.set_marker_text(_marker_text(widget, row))
            card.set_local_status(_status_text(row))
            card.set_expanded(widget._expanded_instance_id() == instance_id)
            _sync_card_enable_visibility(widget, instance_id, row["key"])
            sync_controls_from_status(widget, instance_id)
        except RuntimeError:
            _remove_instance_state(widget, instance_id)
            widget.state.card_widgets.pop(instance_id, None)
            replacement = InstanceCardWidget(
                instance_id=instance_id,
                title=f"{row['name']} [{instance_id}]",
                parent=widget.ui.instances_content,
            )
            replacement.expand_requested.connect(widget._on_card_expand_requested)
            replacement.enable_toggled.connect(widget._on_card_enable_toggled)
            replacement.remove_requested.connect(widget._on_card_remove_requested)
            widget.state.card_widgets[instance_id] = replacement
            widget.state.enable_guard_by_instance[instance_id] = False
            replacement.set_marker_text(_marker_text(widget, row))
            replacement.set_local_status(_status_text(row))
            replacement.set_expanded(widget._expanded_instance_id() == instance_id)
            _sync_card_enable_visibility(widget, instance_id, row["key"])
            sync_controls_from_status(widget, instance_id)

    _rebuild_cards_layout(widget)

    expanded = widget._expanded_instance_id()
    if expanded and expanded not in new_ids:
        widget._set_expanded_instance_id(None)


def _rebuild_cards_layout(widget: OptoControlManagerWidget) -> None:
    layout = widget.instances_layout
    _detach_layout_items(layout)
    for instance_id in sorted(widget.state.card_widgets.keys()):
        layout.addWidget(widget.state.card_widgets[instance_id])
    layout.addStretch(1)


def _marker_text(widget: OptoControlManagerWidget, row: dict[str, Any]) -> str:
    if not is_instance_compatible(widget, row["key"]):
        return "incompatible with modality"
    return ""


def _status_text(row: dict[str, Any]) -> str:
    status = row.get("status", {})
    if isinstance(status, dict):
        return f"Status: {status.get('last_action', 'idle')}"
    return "Status: idle"


def is_instance_compatible(widget: OptoControlManagerWidget, instance_key: str) -> bool:
    contract = widget.modality_service.get_selected_contract()
    if not contract:
        return True

    allowed = contract.get("allowed_optocontrols", [])
    if not allowed:
        return False

    cls = opto_control_registry.get_class(instance_key)
    for allowed_cls in allowed:
        if isinstance(allowed_cls, type) and issubclass(cls, allowed_cls):
            return True
    return False


def _sync_card_enable_visibility(widget: OptoControlManagerWidget, instance_id: str, key: str) -> None:
    card = widget._card_for(instance_id)
    if card is None:
        return
    cls = opto_control_registry.get_class(key)
    methods = {action.method_name for action in getattr(cls, "ACTIONS", [])}
    visible = "enable_mask" in methods and "disable_mask" in methods
    card.set_enable_visible(visible)


def _build_config_form(
    widget: OptoControlManagerWidget,
    parent: QWidget,
    instance_id: str,
    parameter_groups: dict[str, list[Parameter]],
) -> QWidget:
    widget.state.config_widgets_by_instance.setdefault(instance_id, {})
    widget.state.config_params_by_instance.setdefault(instance_id, {})
    widget.state.config_widgets_by_instance[instance_id].clear()
    widget.state.config_params_by_instance[instance_id].clear()

    form_wrap = QWidget(parent)
    config_form = QFormLayout(form_wrap)
    config_form.setContentsMargins(0, 0, 0, 0)
    config_form.setSpacing(8)

    for parameters in parameter_groups.values():
        for param in parameters:
            editor = make_editor(param, form_wrap)
            widget.state.config_widgets_by_instance[instance_id][param.label] = editor
            widget.state.config_params_by_instance[instance_id][param.label] = param

            if param.label == "Mask Path" and isinstance(editor, QLineEdit):
                widget.state.mask_path_widgets_by_instance[instance_id] = editor
                row_widget = QWidget(form_wrap)
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)
                row_layout.addWidget(editor, 1)

                browse_btn = QToolButton(row_widget)
                browse_btn.setIcon(form_wrap.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
                browse_btn.setToolTip("Browse for mask file")
                browse_btn.clicked.connect(
                    lambda _checked=False, target=editor, owner=widget: _on_browse_mask_path(owner, target)
                )
                row_layout.addWidget(browse_btn)

                create_btn = QToolButton(row_widget)
                create_btn.setText("+")
                create_btn.setToolTip("Create mask")
                create_btn.clicked.connect(
                    lambda _checked=False, iid=instance_id, owner=widget: on_mask_editor_open_requested(owner, iid)
                )
                row_layout.addWidget(create_btn)
                widget.state.mask_create_buttons_by_instance[instance_id] = create_btn

                config_form.addRow(param.label, row_widget)
            else:
                config_form.addRow(param.label, editor)
    return form_wrap


def _filtered_actions(instance_key: str, actions: list[Action]) -> list[Action]:
    if instance_key != "mask":
        return actions
    skip = {"load_mask_image", "create_mask", "enable_mask", "disable_mask"}
    return [action for action in actions if action.method_name not in skip]


def _build_actions_area(
    widget: OptoControlManagerWidget,
    parent: QWidget,
    instance_id: str,
    instance_key: str,
    actions: list[Action],
) -> QWidget:
    root = QWidget(parent)
    layout = QVBoxLayout(root)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    filtered_actions = _filtered_actions(instance_key, actions)
    widget.state.action_widgets_by_instance[instance_id] = {}
    widget.state.actions_by_label_by_instance[instance_id] = {action.label: action for action in filtered_actions}
    widget.state.method_to_action_by_instance[instance_id] = {
        action.method_name: action for action in filtered_actions
    }

    for action in filtered_actions:
        action_box = QWidget(root)
        action_layout = QVBoxLayout(action_box)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(6)

        form = QFormLayout()
        action_param_widgets: dict[str, QWidget] = {}
        for param in action.parameters:
            editor = make_editor(param, action_box)
            action_param_widgets[param.label] = editor
            form.addRow(param.label, editor)
        action_layout.addLayout(form)

        button = QPushButton(action.label, action_box)
        button.clicked.connect(
            lambda _checked=False, iid=instance_id, action_label=action.label: run_action(widget, iid, action_label)
        )
        action_layout.addWidget(button)
        widget.state.action_widgets_by_instance[instance_id][action.label] = action_param_widgets
        layout.addWidget(action_box)
    return root


def build_instance_body(widget: OptoControlManagerWidget, instance_id: str, key: str, cls: type) -> QWidget:
    body = QWidget(widget.ui.instances_content)
    layout = QVBoxLayout(body)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    layout.addWidget(_build_config_form(widget, body, instance_id, getattr(cls, "CONFIG_PARAMETERS", {})))
    layout.addWidget(_build_actions_area(widget, body, instance_id, key, getattr(cls, "ACTIONS", [])))
    return body


def _set_expanded_card(widget: OptoControlManagerWidget, instance_id: str | None) -> None:
    current = widget._expanded_instance_id()
    if current and current in widget.state.card_widgets and current != instance_id:
        old_card = widget.state.card_widgets[current]
        _clear_instance_form_state(widget, current)
        old_card.set_expanded(False)
        old_card.set_body_widget(None)
    widget._set_expanded_instance_id(instance_id)


def on_card_expand_requested(widget: OptoControlManagerWidget, instance_id: str) -> None:
    if widget.state.ui_locked_for_editor:
        widget.status_label.setText("Status: close/discard active mask editor first")
        return

    if widget._expanded_instance_id() == instance_id:
        _set_expanded_card(widget, None)
        return

    try:
        key = widget.opto_control_service.get_instance_key(instance_id)
        cls = opto_control_registry.get_class(key)
        _set_expanded_card(widget, instance_id)
        card = widget._card_for(instance_id)
        if card is None:
            return
        card.set_body_widget(build_instance_body(widget, instance_id, key, cls))
        card.set_expanded(True)
        sync_controls_from_status(widget, instance_id)
    except Exception as exc:
        show_error(widget, str(exc), instance_id=instance_id)


def collect_config_values(widget: OptoControlManagerWidget, instance_id: str) -> dict[str, Any]:
    raw_values = collect_values(widget.state.config_widgets_by_instance.get(instance_id, {}))
    values: dict[str, Any] = {}
    for label, raw_value in raw_values.items():
        param = widget.state.config_params_by_instance.get(instance_id, {}).get(label)
        if param is not None and param.param_type is Path and isinstance(raw_value, str):
            stripped = raw_value.strip()
            values[label] = None if stripped == "" else stripped
        else:
            values[label] = raw_value
    return values


def ensure_configured(widget: OptoControlManagerWidget, instance_id: str) -> None:
    widget.opto_control_service.connect(instance_id, collect_config_values(widget, instance_id))


def on_add_clicked(widget: OptoControlManagerWidget) -> None:
    if widget.state.ui_locked_for_editor:
        widget.status_label.setText("Status: close/discard active mask editor first")
        return

    key = widget._selected_type_key()
    if not key:
        return
    try:
        instance_id, _ = widget.opto_control_service.create_opto_control(key)
        widget.status_label.setText(f"Status: added {instance_id}")
        refresh_instances(widget)
        on_card_expand_requested(widget, instance_id)
    except Exception as exc:
        show_error(widget, str(exc))


def on_card_enable_toggled(widget: OptoControlManagerWidget, instance_id: str, checked: bool) -> None:
    if widget.state.enable_guard_by_instance.get(instance_id, False):
        return
    try:
        key = widget.opto_control_service.get_instance_key(instance_id)
        cls = opto_control_registry.get_class(key)
    except Exception:
        return
    method_name = "enable_mask" if checked else "disable_mask"
    action = next((a for a in getattr(cls, "ACTIONS", []) if a.method_name == method_name), None)
    if action is None:
        return
    try:
        ensure_configured(widget, instance_id)
        widget.opto_control_service.run_action(instance_id, action.label, {})
        sync_controls_from_status(widget, instance_id)
    except Exception as exc:
        show_error(widget, str(exc), instance_id=instance_id)
        sync_controls_from_status(widget, instance_id)


def on_card_remove_requested(widget: OptoControlManagerWidget, instance_id: str) -> None:
    if widget.state.ui_locked_for_editor:
        widget.status_label.setText("Status: close/discard active mask editor first")
        return

    try:
        widget.opto_control_service.remove_opto_control(instance_id)
        _cleanup_temp_mask_file(widget, instance_id)
        widget.status_label.setText(f"Status: removed {instance_id}")
    except Exception as exc:
        show_error(widget, str(exc), instance_id=instance_id)


def run_action(widget: OptoControlManagerWidget, instance_id: str, action_label: str) -> None:
    action = widget.state.actions_by_label_by_instance.get(instance_id, {}).get(action_label)
    if action is None:
        show_error(widget, f"Unknown action '{action_label}'", instance_id=instance_id)
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

    raw_args = collect_values(widget.state.action_widgets_by_instance.get(instance_id, {}).get(action_label, {}))
    try:
        ensure_configured(widget, instance_id)
        widget.opto_control_service.run_action(instance_id, action_label, raw_args)
        widget.status_label.setText(f"Status: ran '{action_label}' on {instance_id}")
        sync_controls_from_status(widget, instance_id)
    except Exception as exc:
        show_error(widget, str(exc), instance_id=instance_id)


def sync_controls_from_status(widget: OptoControlManagerWidget, instance_id: str) -> None:
    card = widget._card_for(instance_id)
    if card is None:
        return
    try:
        status = widget.opto_control_service.get_instance(instance_id).get_status()
    except Exception:
        return
    if not isinstance(status, dict):
        return

    daq_widget = widget.state.config_widgets_by_instance.get(instance_id, {}).get("DAQ DO Channel")
    if isinstance(daq_widget, QLineEdit):
        value = status.get("daq_do_channel")
        if value:
            daq_widget.setText(str(value))

    path_widget = widget.state.config_widgets_by_instance.get(instance_id, {}).get("Mask Path")
    if isinstance(path_widget, QLineEdit):
        raw_path = status.get("mask_path")
        path_widget.setText("" if raw_path is None else str(raw_path))

    widget.state.enable_guard_by_instance[instance_id] = True
    card.set_enable_checked(bool(status.get("enabled", False)), guarded=False)
    widget.state.enable_guard_by_instance[instance_id] = False
    card.set_local_status(f"Status: {status.get('last_action', 'idle')}")


def _set_editor_lock(widget: OptoControlManagerWidget, locked: bool) -> None:
    widget.state.ui_locked_for_editor = locked
    widget.type_combo.setEnabled(not locked)
    widget.add_btn.setEnabled(not locked)
    widget.ui.instances_scroll.setEnabled(not locked)


def _on_browse_mask_path(widget: OptoControlManagerWidget, editor: QLineEdit) -> None:
    path, _ = QFileDialog.getOpenFileName(
        widget,
        "Select Mask Image",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
    )
    if not path:
        return
    editor.setText(str(Path(path)))


def on_mask_editor_open_requested(widget: OptoControlManagerWidget, instance_id: str) -> None:
    if widget.state.active_mask_editor_widget is not None:
        widget.status_label.setText("Status: close/discard active mask editor first")
        return

    try:
        key = widget.opto_control_service.get_instance_key(instance_id)
    except Exception as exc:
        show_error(widget, str(exc), instance_id=instance_id)
        return
    if key != "mask":
        show_error(widget, "mask editor is only available for Mask Opto-Control", instance_id=instance_id)
        return

    editor = MaskEditorWidget(parent=widget.ui.editor_host_box)
    editor.create_mask_requested.connect(
        lambda mask, iid=instance_id, owner=widget: on_mask_editor_create_requested(owner, iid, mask)
    )
    editor.cancel_requested.connect(lambda iid=instance_id, owner=widget: on_mask_editor_cancel_requested(owner, iid))

    _mount_editor(widget, editor, instance_id)
    _set_editor_lock(widget, True)
    widget.status_label.setText(f"Status: mask editor open for {instance_id}")


def _close_mask_editor(widget: OptoControlManagerWidget, force_discard: bool = False) -> bool:
    editor = widget.state.active_mask_editor_widget
    if editor is None:
        return True
    if force_discard:
        _unmount_editor(widget)
        _set_editor_lock(widget, False)
        widget.status_label.setText("Status: mask editor closed")
        return True
    if not isinstance(editor, MaskEditorWidget):
        _unmount_editor(widget)
        _set_editor_lock(widget, False)
        return True
    if not editor.is_dirty():
        _unmount_editor(widget)
        _set_editor_lock(widget, False)
        widget.status_label.setText("Status: mask editor closed")
        return True

    prompt = QMessageBox(widget)
    prompt.setIcon(QMessageBox.Icon.Question)
    prompt.setWindowTitle("Unsaved Mask")
    prompt.setText("You have unsaved mask edits.")
    prompt.setInformativeText("Create mask now, discard edits, or cancel closing?")
    create_btn = prompt.addButton("Create Mask", QMessageBox.ButtonRole.AcceptRole)
    discard_btn = prompt.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
    cancel_btn = prompt.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    prompt.setDefaultButton(create_btn)
    prompt.exec()
    clicked = prompt.clickedButton()
    if clicked == create_btn:
        mask = editor.generate_mask()
        if mask is None:
            QMessageBox.warning(widget, "No ROI", "Draw at least one ROI before creating a mask.")
            return False
        on_mask_editor_create_requested(widget, widget.state.active_mask_editor_instance_id or "", mask)
        return widget.state.active_mask_editor_widget is None
    if clicked == discard_btn:
        _unmount_editor(widget)
        _set_editor_lock(widget, False)
        widget.status_label.setText("Status: discarded mask editor changes")
        return True
    if clicked == cancel_btn:
        return False
    return False


def on_mask_editor_cancel_requested(widget: OptoControlManagerWidget, instance_id: str) -> None:
    if widget.state.active_mask_editor_instance_id != instance_id:
        return
    _close_mask_editor(widget, force_discard=False)


def on_mask_editor_create_requested(
    widget: OptoControlManagerWidget,
    instance_id: str,
    mask: np.ndarray,
) -> None:
    if widget.state.active_mask_editor_instance_id != instance_id:
        return
    try:
        ensure_configured(widget, instance_id)
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be a numpy array")
        if mask.ndim != 2:
            raise ValueError("mask must be 2D")
        mask_u8 = (mask > 0).astype(np.uint8) * 255

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_path = Path(temp_file.name)
        temp_file.close()
        write_ok = cv2.imwrite(str(temp_path), mask_u8)
        if not write_ok:
            raise RuntimeError(f"failed to write mask image to '{temp_path}'")

        old_path = widget.state.mask_tempfile_by_instance.get(instance_id)
        widget.state.mask_tempfile_by_instance[instance_id] = temp_path
        if old_path is not None and old_path != temp_path:
            try:
                os.remove(old_path)
            except OSError:
                pass

        widget.opto_control_service.set_mask_data(instance_id, mask_u8, source_path=str(temp_path))
        path_widget = widget.state.mask_path_widgets_by_instance.get(instance_id)
        if path_widget is not None:
            path_widget.setText(str(temp_path))

        sync_controls_from_status(widget, instance_id)
        _unmount_editor(widget)
        _set_editor_lock(widget, False)
        widget.status_label.setText(f"Status: created mask for {instance_id}")
    except Exception as exc:
        show_error(widget, str(exc), instance_id=instance_id)


def on_modality_selected(widget: OptoControlManagerWidget, key: str) -> None:
    del key
    refresh_instances(widget)


def show_error(widget: OptoControlManagerWidget, message: str, instance_id: str | None = None) -> None:
    widget.status_label.setText(f"Status: error - {message}")
    if instance_id:
        card = widget._card_for(instance_id)
        if card is not None:
            card.set_local_status(f"Status: error ({message})")
    QMessageBox.critical(widget, "Opto-Control Error", message)
