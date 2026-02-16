from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.backend_utils.contracts import Action, Parameter
from pyrpoc.optocontrols.opto_control_registry import opto_control_registry
from pyrpoc.services.modality_service import ModalityService
from pyrpoc.services.opto_control_service import OptoControlService


class OptoControlManagerWidget(QWidget):
    def __init__(
        self,
        opto_control_service: OptoControlService,
        modality_service: ModalityService,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.opto_control_service = opto_control_service
        self.modality_service = modality_service

        self._config_widgets: dict[str, QWidget] = {}
        self._config_parameters: dict[str, Parameter] = {}
        self._action_widgets: dict[str, dict[str, QWidget]] = {}
        self._actions_by_label: dict[str, Action] = {}
        self._method_to_action: dict[str, Action] = {}
        self._enable_checkbox: QCheckBox | None = None
        self._enable_guard = False

        self._build_ui()
        self._wire_signals()
        self._refresh_available()
        self._refresh_instances()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Opto-Control:", self))
        self.type_combo = QComboBox(self)
        add_row.addWidget(self.type_combo, 1)
        self.add_btn = QPushButton("Add", self)
        add_row.addWidget(self.add_btn)
        root.addLayout(add_row)

        root.addWidget(QLabel("Instances:", self))
        self.instances_list = QListWidget(self)
        root.addWidget(self.instances_list)

        self.status_label = QLabel("Status: ready", self)
        root.addWidget(self.status_label)

        self.controls_box = QGroupBox("Opto-Control", self)
        self.controls_layout = QVBoxLayout(self.controls_box)
        self.controls_layout.setContentsMargins(8, 8, 8, 8)
        self.controls_layout.setSpacing(8)
        root.addWidget(self.controls_box, 1)

    def _wire_signals(self) -> None:
        self.add_btn.clicked.connect(self._on_add_clicked)
        self.instances_list.currentItemChanged.connect(self._on_instance_selected)

        self.opto_control_service.inventory_changed.connect(self._refresh_instances)
        self.opto_control_service.connection_changed.connect(self._on_connection_changed)
        self.modality_service.modality_selected.connect(self._on_modality_selected)

    def _refresh_available(self) -> None:
        current_key = self._selected_type_key()
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        for row in self.opto_control_service.list_available():
            key = row["key"]
            name = row.get("display_name", key)
            self.type_combo.addItem(name, key)
        self.type_combo.blockSignals(False)

        if current_key:
            idx = self.type_combo.findData(current_key)
            if idx >= 0:
                self.type_combo.setCurrentIndex(idx)
        elif self.type_combo.count() > 0:
            self.type_combo.setCurrentIndex(0)

    def _refresh_instances(self) -> None:
        previous = self._selected_instance_id()
        selected_found = False
        self.instances_list.blockSignals(True)
        self.instances_list.clear()
        for row in self.opto_control_service.list_instances():
            status = row.get("status", {})
            if isinstance(status, dict) and "enabled" in status:
                marker = "enabled" if bool(status["enabled"]) else "disabled"
            else:
                marker = "connected" if row["connected"] else "disconnected"
            compat_label = ""
            if not self._is_instance_compatible(row["key"]):
                compat_label = ", incompatible with current modality"
            item = QListWidgetItem(
                f"{row['name']} [{row['instance_id']}] ({marker}{compat_label})"
            )
            item.setData(Qt.ItemDataRole.UserRole, row["instance_id"])
            self.instances_list.addItem(item)
        self.instances_list.blockSignals(False)

        if previous:
            for idx in range(self.instances_list.count()):
                item = self.instances_list.item(idx)
                if item.data(Qt.ItemDataRole.UserRole) == previous:
                    self.instances_list.setCurrentRow(idx)
                    selected_found = True
                    break
        if not selected_found:
            self._clear_dynamic_panels()

    def _is_instance_compatible(self, instance_key: str) -> bool:
        contract = self.modality_service.get_selected_contract()
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

    def _clear_dynamic_panels(self) -> None:
        self._config_widgets.clear()
        self._config_parameters.clear()
        self._action_widgets.clear()
        self._actions_by_label.clear()
        self._method_to_action.clear()
        self._enable_checkbox = None
        while self.controls_layout.count():
            item = self.controls_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.controls_layout.addWidget(QLabel("Select an opto-control instance.", self.controls_box))
        self.controls_layout.addStretch(1)

    def _build_config_form(self, parameter_groups: dict[str, list[Parameter]]) -> None:
        self._config_widgets.clear()
        self._config_parameters.clear()
        config_box = QGroupBox("Parameters", self.controls_box)
        config_form = QFormLayout(config_box)

        for group_name, parameters in parameter_groups.items():
            if group_name:
                header = QLabel(f"[{group_name.capitalize()}]", config_box)
                header.setStyleSheet("font-weight: bold;")
                config_form.addRow(header)
            for param in parameters:
                editor = self._make_editor(param, config_box)
                self._config_widgets[param.label] = editor
                self._config_parameters[param.label] = param
                config_form.addRow(param.label, editor)
        self.controls_layout.addWidget(config_box)

    def _build_actions_area(self, actions: list[Action]) -> None:
        self._action_widgets.clear()
        self._actions_by_label = {action.label: action for action in actions}
        self._method_to_action = {action.method_name: action for action in actions}

        for action in actions:
            action_box = QGroupBox(action.label, self.controls_box)
            box_layout = QVBoxLayout(action_box)

            if action.tooltip:
                action_box.setToolTip(action.tooltip)

            form = QFormLayout()
            action_param_widgets: dict[str, QWidget] = {}
            for param in action.parameters:
                editor = self._make_editor(param, action_box)
                action_param_widgets[param.label] = editor
                form.addRow(param.label, editor)

            box_layout.addLayout(form)

            run_btn = QPushButton("Run", action_box)
            run_btn.clicked.connect(
                lambda checked, action_label=action.label: self._run_action(action_label)
            )
            box_layout.addWidget(run_btn)

            self._action_widgets[action.label] = action_param_widgets
            self.controls_layout.addWidget(action_box)

    def _build_mask_controls(self, actions: list[Action]) -> None:
        self._actions_by_label = {}
        self._method_to_action = {}
        for action in actions:
            self._actions_by_label[action.label] = action
            self._method_to_action[action.method_name] = action

        button_row = QHBoxLayout()

        load_action = self._method_to_action.get("load_mask_image")
        if load_action is not None:
            load_btn = QPushButton("Load Mask", self.controls_box)
            load_btn.clicked.connect(lambda: self._run_action(load_action.label))
            button_row.addWidget(load_btn)

        create_action = self._method_to_action.get("create_mask")
        if create_action is not None:
            create_btn = QPushButton("Create Mask", self.controls_box)
            create_btn.clicked.connect(lambda: self._run_action(create_action.label))
            button_row.addWidget(create_btn)

        button_row.addStretch(1)
        button_wrap = QWidget(self.controls_box)
        button_wrap.setLayout(button_row)
        self.controls_layout.addWidget(button_wrap)

        if "enable_mask" in self._method_to_action and "disable_mask" in self._method_to_action:
            self._enable_checkbox = QCheckBox("Enable", self.controls_box)
            self._enable_checkbox.toggled.connect(self._on_enable_toggled)
            self.controls_layout.addWidget(self._enable_checkbox)

    def _build_control_panel(self, key: str, cls: type) -> None:
        self._actions_by_label = {}
        self._method_to_action = {}
        self._clear_dynamic_panels()
        while self.controls_layout.count():
            item = self.controls_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._build_config_form(getattr(cls, "CONFIG_PARAMETERS", {}))
        if key == "mask":
            self._build_mask_controls(getattr(cls, "ACTIONS", []))
        else:
            self._build_actions_area(getattr(cls, "ACTIONS", []))

        self.controls_layout.addStretch(1)

    def _collect_config_values(self) -> dict[str, Any]:
        raw_values = self._collect_values(self._config_widgets)
        values: dict[str, Any] = {}
        for label, raw_value in raw_values.items():
            param = self._config_parameters.get(label)
            if param is not None and param.param_type is Path and isinstance(raw_value, str):
                stripped = raw_value.strip()
                values[label] = None if stripped == "" else stripped
            else:
                values[label] = raw_value
        return values

    def _make_editor(self, param: Parameter, parent: QWidget) -> QWidget:
        default = param.default

        if param.param_type is int:
            editor = QSpinBox(parent)
            editor.setMinimum(int(param.minimum if param.minimum is not None else -1_000_000))
            editor.setMaximum(int(param.maximum if param.maximum is not None else 1_000_000))
            editor.setValue(int(default if default is not None else 0))
        elif param.param_type is float:
            editor = QDoubleSpinBox(parent)
            editor.setDecimals(6)
            editor.setMinimum(float(param.minimum if param.minimum is not None else -1e12))
            editor.setMaximum(float(param.maximum if param.maximum is not None else 1e12))
            editor.setSingleStep(float(param.step if param.step is not None else 0.1))
            editor.setValue(float(default if default is not None else 0.0))
        elif param.param_type is bool:
            editor = QCheckBox(parent)
            editor.setChecked(bool(default))
        elif param.param_type is str and param.choices:
            editor = QComboBox(parent)
            editor.addItems([str(v) for v in param.choices])
            if default is not None:
                editor.setCurrentText(str(default))
        else:
            editor = QLineEdit(parent)
            editor.setText("" if default is None else str(default))
            if param.param_type is Path:
                editor.setPlaceholderText("Path")

        if param.tooltip:
            editor.setToolTip(param.tooltip)
        return editor

    def _collect_values(self, widget_map: dict[str, QWidget]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for label, widget in widget_map.items():
            if isinstance(widget, QSpinBox):
                values[label] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                values[label] = widget.value()
            elif isinstance(widget, QCheckBox):
                values[label] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[label] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                values[label] = widget.text()
        return values

    def _ensure_configured(self, instance_id: str) -> None:
        config = self._collect_config_values()
        self.opto_control_service.connect(instance_id, config)

    def _on_add_clicked(self) -> None:
        key = self._selected_type_key()
        if not key:
            return

        try:
            instance_id, _ = self.opto_control_service.create_opto_control(key)
            self.status_label.setText(f"Status: added {instance_id}")
            self._refresh_instances()
        except Exception as exc:
            self._show_error(str(exc))

    def _on_enable_toggled(self, checked: bool) -> None:
        if self._enable_guard:
            return
        instance_id = self._selected_instance_id()
        if not instance_id:
            self._show_error("Select an opto-control instance first")
            return
        try:
            self._ensure_configured(instance_id)
            action_method = "enable_mask" if checked else "disable_mask"
            action = self._method_to_action.get(action_method)
            if action is None:
                return
            self.opto_control_service.run_action(instance_id, action.label, {})
            self._sync_controls_from_status(instance_id)
        except Exception as exc:
            self._show_error(str(exc))
            self._sync_controls_from_status(instance_id)

    def _run_action(self, action_label: str) -> None:
        instance_id = self._selected_instance_id()
        if not instance_id:
            self._show_error("Select an opto-control instance first")
            return

        action = self._actions_by_label.get(action_label)
        if action is None:
            self._show_error(f"Unknown action '{action_label}'")
            return

        if action.dangerous:
            prompt = action.confirm_text or f"Run action '{action.label}'?"
            response = QMessageBox.question(
                self,
                "Confirm Action",
                prompt,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return

        raw_args = self._collect_values(self._action_widgets.get(action_label, {}))
        try:
            self._ensure_configured(instance_id)
            self.opto_control_service.run_action(instance_id, action_label, raw_args)
            self.status_label.setText(f"Status: ran '{action_label}' on {instance_id}")
            self._sync_controls_from_status(instance_id)
        except Exception as exc:
            self._show_error(str(exc))

    def _on_instance_selected(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        del previous
        if current is None:
            self._clear_dynamic_panels()
            return

        instance_id = self._selected_instance_id()
        if not instance_id:
            self._clear_dynamic_panels()
            return

        try:
            key = self.opto_control_service.get_instance_key(instance_id)
            cls = opto_control_registry.get_class(key)
            self._build_control_panel(key, cls)
            self._sync_controls_from_status(instance_id)
        except Exception:
            self._clear_dynamic_panels()

    def _sync_controls_from_status(self, instance_id: str) -> None:
        try:
            status = self.opto_control_service.get_instance(instance_id).get_status()
        except Exception:
            return
        if not isinstance(status, dict):
            return

        daq_widget = self._config_widgets.get("DAQ DO Channel")
        if isinstance(daq_widget, QLineEdit):
            value = status.get("daq_do_channel")
            if value:
                daq_widget.setText(str(value))

        path_widget = self._config_widgets.get("Mask Path")
        if isinstance(path_widget, QLineEdit):
            raw_path = status.get("mask_path")
            path_widget.setText("" if raw_path is None else str(raw_path))

        if self._enable_checkbox is not None:
            self._enable_guard = True
            self._enable_checkbox.setChecked(bool(status.get("enabled", False)))
            self._enable_guard = False

    def _on_connection_changed(self, instance_id: str, connected: bool) -> None:
        state = "connected" if connected else "disconnected"
        self.status_label.setText(f"Status: {instance_id} {state}")
        self._refresh_instances()
        selected = self._selected_instance_id()
        if selected == instance_id:
            self._sync_controls_from_status(instance_id)

    def _on_modality_selected(self, key: str) -> None:
        del key
        self._refresh_instances()

    def _selected_type_key(self) -> str:
        data = self.type_combo.currentData()
        if isinstance(data, str):
            return data
        return self.type_combo.currentText().strip()

    def _selected_instance_id(self) -> str:
        item = self.instances_list.currentItem()
        if item is None:
            return ""
        value = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(value, str):
            return value
        return ""

    def _show_error(self, message: str) -> None:
        self.status_label.setText(f"Status: error - {message}")
        QMessageBox.critical(self, "Opto-Control Error", message)
