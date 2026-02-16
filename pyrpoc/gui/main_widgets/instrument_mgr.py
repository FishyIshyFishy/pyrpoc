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
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.backend_utils.contracts import Action, Parameter
from pyrpoc.instruments.instrument_registry import instrument_registry
from pyrpoc.services.instrument_service import InstrumentService


class InstrumentManagerWidget(QWidget):
    def __init__(self, instrument_service: InstrumentService, parent: QWidget | None = None):
        super().__init__(parent)
        self.instrument_service = instrument_service

        self._config_widgets: dict[str, QWidget] = {}
        self._action_widgets: dict[str, dict[str, QWidget]] = {}
        self._actions_by_label: dict[str, Action] = {}

        self._build_ui()
        self._wire_signals()
        self._refresh_available()
        self._refresh_instances()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Instrument:", self))
        self.type_combo = QComboBox(self)
        add_row.addWidget(self.type_combo, 1)
        self.add_btn = QPushButton("Add", self)
        add_row.addWidget(self.add_btn)
        root.addLayout(add_row)

        root.addWidget(QLabel("Instances:", self))
        self.instances_list = QListWidget(self)
        root.addWidget(self.instances_list)

        action_row = QHBoxLayout()
        self.connect_btn = QPushButton("Connect", self)
        self.disconnect_btn = QPushButton("Disconnect", self)
        self.remove_btn = QPushButton("Remove", self)
        action_row.addWidget(self.connect_btn)
        action_row.addWidget(self.disconnect_btn)
        action_row.addWidget(self.remove_btn)
        action_row.addStretch(1)
        root.addLayout(action_row)

        self.status_label = QLabel("Status: ready", self)
        root.addWidget(self.status_label)

        self.config_box = QGroupBox("Connection Parameters", self)
        self.config_form = QFormLayout(self.config_box)
        root.addWidget(self.config_box)

        self.actions_box = QGroupBox("Actions", self)
        self.actions_layout = QVBoxLayout(self.actions_box)
        self.actions_layout.setContentsMargins(8, 8, 8, 8)
        self.actions_layout.setSpacing(8)
        self.actions_layout.addStretch(1)

        actions_scroll = QScrollArea(self)
        actions_scroll.setWidgetResizable(True)
        actions_scroll.setWidget(self.actions_box)
        root.addWidget(actions_scroll, 1)

    def _wire_signals(self) -> None:
        self.add_btn.clicked.connect(self._on_add_clicked)
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        self.disconnect_btn.clicked.connect(self._on_disconnect_clicked)
        self.remove_btn.clicked.connect(self._on_remove_clicked)
        self.instances_list.currentItemChanged.connect(self._on_instance_selected)

        self.instrument_service.inventory_changed.connect(self._refresh_instances)
        self.instrument_service.connection_changed.connect(self._on_connection_changed)

    def _refresh_available(self) -> None:
        current_key = self._selected_type_key()
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        for row in self.instrument_service.list_available():
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
        for row in self.instrument_service.list_instances():
            marker = "connected" if row["connected"] else "disconnected"
            item = QListWidgetItem(f"{row['name']} [{row['instance_id']}] ({marker})")
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

    def _clear_dynamic_panels(self) -> None:
        self._build_config_form({})
        self._build_actions_area([])

    def _build_config_form(self, parameter_groups: dict[str, list[Parameter]]) -> None:
        self._config_widgets.clear()
        while self.config_form.rowCount() > 0:
            self.config_form.removeRow(0)

        for group_name, parameters in parameter_groups.items():
            if group_name:
                header = QLabel(f"[{group_name.capitalize()}]", self.config_box)
                header.setStyleSheet("font-weight: bold;")
                self.config_form.addRow(header)
            for param in parameters:
                editor = self._make_editor(param, self.config_box)
                self._config_widgets[param.label] = editor
                self.config_form.addRow(param.label, editor)

    def _build_actions_area(self, actions: list[Action]) -> None:
        self._action_widgets.clear()
        self._actions_by_label = {action.label: action for action in actions}

        while self.actions_layout.count():
            item = self.actions_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for action in actions:
            action_box = QGroupBox(action.label, self.actions_box)
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
            run_btn.clicked.connect(lambda checked, action_label=action.label: self._run_action(action_label))
            box_layout.addWidget(run_btn)

            self._action_widgets[action.label] = action_param_widgets
            self.actions_layout.addWidget(action_box)

        self.actions_layout.addStretch(1)

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

    def _on_add_clicked(self) -> None:
        key = self._selected_type_key()
        if not key:
            return

        try:
            instance_id, _ = self.instrument_service.create_instrument(key)
            self.status_label.setText(f"Status: added {instance_id}")
            self._refresh_instances()
        except Exception as exc:
            self._show_error(str(exc))

    def _on_connect_clicked(self) -> None:
        instance_id = self._selected_instance_id()
        if not instance_id:
            self._show_error("Select an instrument instance first")
            return
        try:
            config = self._collect_values(self._config_widgets)
            self.instrument_service.connect(instance_id, config)
        except Exception as exc:
            self._show_error(str(exc))

    def _on_disconnect_clicked(self) -> None:
        instance_id = self._selected_instance_id()
        if not instance_id:
            self._show_error("Select an instrument instance first")
            return
        try:
            self.instrument_service.disconnect(instance_id)
        except Exception as exc:
            self._show_error(str(exc))

    def _on_remove_clicked(self) -> None:
        instance_id = self._selected_instance_id()
        if not instance_id:
            return
        self.instrument_service.remove_instrument(instance_id)
        self.status_label.setText(f"Status: removed {instance_id}")

    def _run_action(self, action_label: str) -> None:
        instance_id = self._selected_instance_id()
        if not instance_id:
            self._show_error("Select an instrument instance first")
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
            self.instrument_service.run_action(instance_id, action_label, raw_args)
            self.status_label.setText(f"Status: ran '{action_label}' on {instance_id}")
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
            key = self.instrument_service.get_instance_key(instance_id)
            cls = instrument_registry.get_class(key)
            self._build_config_form(cls.CONFIG_PARAMETERS)
            self._build_actions_area(cls.ACTIONS)
        except Exception:
            self._clear_dynamic_panels()

    def _on_connection_changed(self, instance_id: str, connected: bool) -> None:
        state = "connected" if connected else "disconnected"
        self.status_label.setText(f"Status: {instance_id} {state}")
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
        QMessageBox.critical(self, "Instrument Error", message)
