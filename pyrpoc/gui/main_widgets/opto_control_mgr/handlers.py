from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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

from pyrpoc.backend_utils.contracts import Action
from pyrpoc.gui.main_widgets.opto_control_mgr.forms import collect_values, make_editor
from pyrpoc.gui.main_widgets.opto_control_mgr.instance_card import InstanceCardWidget
from pyrpoc.gui.main_widgets.opto_control_mgr.mask_editor import MaskEditorWidget
from pyrpoc.services.opto_control_types import OptoInstanceRow, OptoInstanceSchema, OptoStatusSnapshot

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
    placeholder = QLabel("Editor area reserved.", widget.ui.editor_host_box)
    widget.editor_host_layout.addWidget(placeholder)
    widget.editor_host_layout.addStretch(1)


def _mount_editor(widget: OptoControlManagerWidget, editor: QWidget, instance_id: object) -> None:
    _detach_editor_layout_items(widget.editor_host_layout)
    widget.editor_host_layout.addWidget(editor)
    widget.state.active_editor_widget = editor
    widget.state.active_editor_instance = instance_id


def _unmount_editor(widget: OptoControlManagerWidget) -> None:
    editor = widget.state.active_editor_widget
    widget.state.active_editor_widget = None
    widget.state.active_editor_instance = None
    if editor is not None:
        editor.setParent(None)
        editor.deleteLater()
    _set_editor_host_placeholder(widget)


def _detach_layout_items(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        child = item.widget()
        if child is not None:
            child.setParent(None)


def _remove_instance_state(widget: OptoControlManagerWidget, instance_id: object) -> None:
    widget.state.config_widgets_by_instance.pop(instance_id, None)
    widget.state.config_params_by_instance.pop(instance_id, None)
    widget.state.action_widgets_by_instance.pop(instance_id, None)
    widget.state.actions_by_label_by_instance.pop(instance_id, None)
    widget.state.enable_guard_by_instance.pop(instance_id, None)
    widget.state.editor_anchor_widgets_by_instance.pop(instance_id, None)
    widget.state.editor_launch_buttons_by_instance.pop(instance_id, None)


def _clear_instance_form_state(widget: OptoControlManagerWidget, instance_id: object) -> None:
    widget.state.config_widgets_by_instance.pop(instance_id, None)
    widget.state.config_params_by_instance.pop(instance_id, None)
    widget.state.action_widgets_by_instance.pop(instance_id, None)
    widget.state.actions_by_label_by_instance.pop(instance_id, None)
    widget.state.editor_anchor_widgets_by_instance.pop(instance_id, None)
    widget.state.editor_launch_buttons_by_instance.pop(instance_id, None)


def refresh_instances(widget: OptoControlManagerWidget) -> None:
    contract = widget.modality_service.get_selected_contract()
    rows = widget.opto_control_service.list_instance_rows(contract)
    rows_by_id = {row.state: row for row in rows}
    current_ids = set(widget.state.card_widgets.keys())
    new_ids = set(rows_by_id.keys())

    for removed_id in list(current_ids - new_ids):
        if widget.state.active_editor_instance == removed_id:
            _close_editor(widget, force_discard=True)
        card = widget.state.card_widgets.pop(removed_id)
        card.setParent(None)
        card.deleteLater()
        _remove_instance_state(widget, removed_id)
        if widget._expanded_instance() == removed_id:
            widget._set_expanded_instance(None)

    for idx, row in enumerate(rows, start=1):
        instance_id = row.state
        card = widget.state.card_widgets.get(instance_id)
        if card is None:
            card = InstanceCardWidget(
                state_obj=instance_id,
                title=f"{row.display_name} [{idx}]",
                parent=widget.ui.instances_content,
            )
            card.expand_requested.connect(widget._on_card_expand_requested)
            card.enable_toggled.connect(widget._on_card_enable_toggled)
            card.remove_requested.connect(widget._on_card_remove_requested)
            widget.state.card_widgets[instance_id] = card
            widget.state.enable_guard_by_instance[instance_id] = False
        try:
            card.set_marker_text(_marker_text(row))
            card.set_local_status(f"Status: {row.status.last_action}")
            card.set_expanded(widget._expanded_instance() == instance_id)
            card.set_enable_visible(True)
            sync_controls_from_status(widget, instance_id)
        except RuntimeError:
            _remove_instance_state(widget, instance_id)
            widget.state.card_widgets.pop(instance_id, None)
            replacement = InstanceCardWidget(
                state_obj=instance_id,
                title=f"{row.display_name} [{idx}]",
                parent=widget.ui.instances_content,
            )
            replacement.expand_requested.connect(widget._on_card_expand_requested)
            replacement.enable_toggled.connect(widget._on_card_enable_toggled)
            replacement.remove_requested.connect(widget._on_card_remove_requested)
            widget.state.card_widgets[instance_id] = replacement
            widget.state.enable_guard_by_instance[instance_id] = False
            replacement.set_marker_text(_marker_text(row))
            replacement.set_local_status(f"Status: {row.status.last_action}")
            replacement.set_expanded(widget._expanded_instance() == instance_id)
            replacement.set_enable_visible(True)
            sync_controls_from_status(widget, instance_id)

    _rebuild_cards_layout(widget)

    expanded = widget._expanded_instance()
    if expanded and expanded not in new_ids:
        widget._set_expanded_instance(None)


def _rebuild_cards_layout(widget: OptoControlManagerWidget) -> None:
    layout = widget.instances_layout
    _detach_layout_items(layout)
    for instance_id in widget.state.card_widgets.keys():
        layout.addWidget(widget.state.card_widgets[instance_id])
    layout.addStretch(1)


def _marker_text(row: OptoInstanceRow) -> str:
    if not row.compatible_with_selected_modality:
        return "incompatible with modality"
    return ""


def _build_config_form(
    widget: OptoControlManagerWidget,
    parent: QWidget,
    instance_id: object,
    schema: OptoInstanceSchema,
) -> QWidget:
    widget.state.config_widgets_by_instance.setdefault(instance_id, {})
    widget.state.config_params_by_instance.setdefault(instance_id, {})
    widget.state.editor_anchor_widgets_by_instance.setdefault(instance_id, {})
    widget.state.config_widgets_by_instance[instance_id].clear()
    widget.state.config_params_by_instance[instance_id].clear()
    widget.state.editor_anchor_widgets_by_instance[instance_id].clear()

    form_wrap = QWidget(parent)
    config_form = QFormLayout(form_wrap)
    config_form.setContentsMargins(0, 0, 0, 0)
    config_form.setSpacing(8)

    for parameters in schema.config_parameters.values():
        for param in parameters:
            editor = make_editor(param, form_wrap)
            widget.state.config_widgets_by_instance[instance_id][param.label] = editor
            widget.state.config_params_by_instance[instance_id][param.label] = param

            row_widget = QWidget(form_wrap)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            row_layout.addWidget(editor, 1)

            has_aux = False
            if param.param_type is Path and isinstance(editor, QLineEdit):
                browse_btn = QToolButton(row_widget)
                browse_btn.setIcon(form_wrap.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
                browse_btn.setToolTip("Browse for file")
                browse_btn.clicked.connect(
                    lambda _checked=False, target=editor, owner=widget: _on_browse_path(owner, target)
                )
                row_layout.addWidget(browse_btn)
                has_aux = True

            if schema.editor_key and schema.editor_anchor_param == param.label:
                create_btn = QToolButton(row_widget)
                create_btn.setText("+")
                create_btn.setToolTip(f"Open {schema.editor_key} editor")
                create_btn.clicked.connect(
                    lambda _checked=False, iid=instance_id, owner=widget: on_editor_open_requested(owner, iid)
                )
                row_layout.addWidget(create_btn)
                widget.state.editor_launch_buttons_by_instance[instance_id] = create_btn
                widget.state.editor_anchor_widgets_by_instance[instance_id][param.label] = editor
                has_aux = True

            if has_aux:
                config_form.addRow(param.label, row_widget)
            else:
                config_form.addRow(param.label, editor)

    return form_wrap


def _build_actions_area(
    widget: OptoControlManagerWidget,
    parent: QWidget,
    instance_id: object,
    schema: OptoInstanceSchema,
) -> QWidget:
    root = QWidget(parent)
    layout = QVBoxLayout(root)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    widget.state.action_widgets_by_instance[instance_id] = {}
    widget.state.actions_by_label_by_instance[instance_id] = {action.label: action for action in schema.actions}

    for action in schema.actions:
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


def build_instance_body(
    widget: OptoControlManagerWidget,
    instance_id: object,
    schema: OptoInstanceSchema,
) -> QWidget:
    body = QWidget(widget.ui.instances_content)
    layout = QVBoxLayout(body)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    layout.addWidget(_build_config_form(widget, body, instance_id, schema))
    layout.addWidget(_build_actions_area(widget, body, instance_id, schema))
    return body


def _set_expanded_card(widget: OptoControlManagerWidget, instance_id: object | None) -> None:
    current = widget._expanded_instance()
    if current and current in widget.state.card_widgets and current != instance_id:
        old_card = widget.state.card_widgets[current]
        _clear_instance_form_state(widget, current)
        old_card.set_expanded(False)
        old_card.set_body_widget(None)
    widget._set_expanded_instance(instance_id)


def on_card_expand_requested(widget: OptoControlManagerWidget, instance_id: object) -> None:
    if widget.state.ui_locked_for_editor:
        widget.status_label.setText("Status: close/discard active editor first")
        return

    if widget._expanded_instance() == instance_id:
        _set_expanded_card(widget, None)
        return

    try:
        schema = widget.opto_control_service.get_instance_schema(instance_id)
        _set_expanded_card(widget, instance_id)
        card = widget._card_for(instance_id)
        if card is None:
            return
        card.set_body_widget(build_instance_body(widget, instance_id, schema))
        card.set_expanded(True)
        sync_controls_from_status(widget, instance_id)
    except Exception as exc:
        show_error(widget, str(exc), instance_id=instance_id)


def collect_config_values(widget: OptoControlManagerWidget, instance_id: object) -> dict[str, Any]:
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


def on_add_clicked(widget: OptoControlManagerWidget) -> None:
    if widget.state.ui_locked_for_editor:
        widget.status_label.setText("Status: close/discard active editor first")
        return

    key = widget._selected_type_key()
    if not key:
        return
    try:
        instance_id = widget.opto_control_service.create_opto_control(key)
        widget.status_label.setText("Status: added opto-control")
        refresh_instances(widget)
        on_card_expand_requested(widget, instance_id)
    except Exception as exc:
        show_error(widget, str(exc))


def on_card_enable_toggled(widget: OptoControlManagerWidget, instance_id: object, checked: bool) -> None:
    if widget.state.enable_guard_by_instance.get(instance_id, False):
        return
    try:
        raw_config = collect_config_values(widget, instance_id)
        widget.opto_control_service.set_enabled(instance_id, checked, raw_config)
        sync_controls_from_status(widget, instance_id)
    except Exception as exc:
        show_error(widget, str(exc), instance_id=instance_id)
        sync_controls_from_status(widget, instance_id)


def on_card_remove_requested(widget: OptoControlManagerWidget, instance_id: object) -> None:
    if widget.state.ui_locked_for_editor:
        widget.status_label.setText("Status: close/discard active editor first")
        return

    try:
        widget.opto_control_service.remove_opto_control(instance_id)
        widget.status_label.setText("Status: removed opto-control")
    except Exception as exc:
        show_error(widget, str(exc), instance_id=instance_id)


def run_action(widget: OptoControlManagerWidget, instance_id: object, action_label: str) -> None:
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
    raw_config = collect_config_values(widget, instance_id)
    try:
        result = widget.opto_control_service.run_action_with_auto_connect(instance_id, action_label, raw_args, raw_config)
        widget.status_label.setText(f"Status: ran '{result.action_label}'")
        sync_controls_from_status(widget, instance_id)
    except Exception as exc:
        show_error(widget, str(exc), instance_id=instance_id)


def sync_controls_from_status(widget: OptoControlManagerWidget, instance_id: object) -> None:
    card = widget._card_for(instance_id)
    if card is None:
        return
    try:
        status = widget.opto_control_service.get_status_snapshot(instance_id)
    except Exception:
        return

    _apply_status_to_controls(widget, instance_id, status)


def _apply_status_to_controls(
    widget: OptoControlManagerWidget,
    instance_id: object,
    status: OptoStatusSnapshot,
) -> None:
    card = widget._card_for(instance_id)
    if card is None:
        return

    widget.state.enable_guard_by_instance[instance_id] = True
    card.set_enable_checked(status.enabled, guarded=False)
    widget.state.enable_guard_by_instance[instance_id] = False
    card.set_local_status(f"Status: {status.last_action}")


def _set_editor_lock(widget: OptoControlManagerWidget, locked: bool) -> None:
    widget.state.ui_locked_for_editor = locked
    widget.type_combo.setEnabled(not locked)
    widget.add_btn.setEnabled(not locked)
    widget.ui.instances_scroll.setEnabled(not locked)


def _on_browse_path(widget: OptoControlManagerWidget, editor: QLineEdit) -> None:
    path, _ = QFileDialog.getOpenFileName(
        widget,
        "Select File",
        "",
        "All Files (*)",
    )
    if not path:
        return
    editor.setText(str(Path(path)))


def _build_editor_for_key(editor_key: str | None, parent: QWidget) -> QWidget | None:
    if editor_key == "mask":
        return MaskEditorWidget(parent=parent)
    return None


def on_editor_open_requested(widget: OptoControlManagerWidget, instance_id: object) -> None:
    if widget.state.active_editor_widget is not None:
        widget.status_label.setText("Status: close/discard active editor first")
        return

    try:
        schema = widget.opto_control_service.get_instance_schema(instance_id)
    except Exception as exc:
        show_error(widget, str(exc), instance_id=instance_id)
        return
    if not schema.editor_key:
        show_error(widget, "no editor is configured for this opto-control", instance_id=instance_id)
        return

    editor = _build_editor_for_key(schema.editor_key, widget.ui.editor_host_box)
    if editor is None:
        show_error(widget, f"unknown editor key '{schema.editor_key}'", instance_id=instance_id)
        return

    if isinstance(editor, MaskEditorWidget):
        editor.create_mask_requested.connect(
            lambda payload, iid=instance_id, owner=widget: on_editor_create_requested(owner, iid, payload)
        )
        editor.cancel_requested.connect(lambda iid=instance_id, owner=widget: on_editor_cancel_requested(owner, iid))

    _mount_editor(widget, editor, instance_id)
    _set_editor_lock(widget, True)
    widget.status_label.setText(f"Status: editor open ({schema.editor_key})")


def _close_editor(widget: OptoControlManagerWidget, force_discard: bool = False) -> bool:
    editor = widget.state.active_editor_widget
    if editor is None:
        return True
    if force_discard:
        _unmount_editor(widget)
        _set_editor_lock(widget, False)
        widget.status_label.setText("Status: editor closed")
        return True

    is_dirty = bool(getattr(editor, "is_dirty", lambda: False)())
    if not is_dirty:
        _unmount_editor(widget)
        _set_editor_lock(widget, False)
        widget.status_label.setText("Status: editor closed")
        return True

    prompt = QMessageBox(widget)
    prompt.setIcon(QMessageBox.Icon.Question)
    prompt.setWindowTitle("Unsaved Editor State")
    prompt.setText("You have unsaved editor changes.")
    prompt.setInformativeText("Apply now, discard changes, or cancel closing?")
    apply_btn = prompt.addButton("Apply", QMessageBox.ButtonRole.AcceptRole)
    discard_btn = prompt.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
    cancel_btn = prompt.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    prompt.setDefaultButton(apply_btn)
    prompt.exec()
    clicked = prompt.clickedButton()
    if clicked == apply_btn:
        payload = _get_editor_payload_for_apply(editor, widget)
        if payload is None:
            return False
        on_editor_create_requested(widget, widget.state.active_editor_instance, payload)
        return widget.state.active_editor_widget is None
    if clicked == discard_btn:
        _unmount_editor(widget)
        _set_editor_lock(widget, False)
        widget.status_label.setText("Status: discarded editor changes")
        return True
    if clicked == cancel_btn:
        return False
    return False


def _get_editor_payload_for_apply(editor: QWidget, widget: OptoControlManagerWidget) -> object | None:
    if isinstance(editor, MaskEditorWidget):
        payload = editor.generate_mask()
        if payload is None:
            QMessageBox.warning(widget, "No ROI", "Draw at least one ROI before applying.")
            return None
        return payload
    QMessageBox.warning(widget, "Unsupported Editor", "Editor does not support apply payload extraction.")
    return None


def on_editor_cancel_requested(widget: OptoControlManagerWidget, instance_id: object) -> None:
    if widget.state.active_editor_instance != instance_id:
        return
    _close_editor(widget, force_discard=False)


def on_editor_create_requested(
    widget: OptoControlManagerWidget,
    instance_id: object,
    payload: object,
) -> None:
    if widget.state.active_editor_instance != instance_id:
        return
    try:
        raw_config = collect_config_values(widget, instance_id)
        status = widget.opto_control_service.apply_editor_payload(instance_id, payload, raw_config)
        _apply_status_to_controls(widget, instance_id, status)
        _unmount_editor(widget)
        _set_editor_lock(widget, False)
        widget.status_label.setText("Status: editor payload applied")
    except Exception as exc:
        show_error(widget, str(exc), instance_id=instance_id)


def on_modality_selected(widget: OptoControlManagerWidget, key: str) -> None:
    del key
    refresh_instances(widget)


def show_error(widget: OptoControlManagerWidget, message: str, instance_id: object | None = None) -> None:
    widget.status_label.setText(f"Status: error - {message}")
    if instance_id:
        card = widget._card_for(instance_id)
        if card is not None:
            card.set_local_status(f"Status: error ({message})")
    QMessageBox.critical(widget, "Opto-Control Error", message)
