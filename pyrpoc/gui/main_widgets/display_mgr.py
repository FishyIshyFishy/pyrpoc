from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.backend_utils.contracts import Parameter
from pyrpoc.backend_utils.data import BaseData
from pyrpoc.displays.display_registry import display_registry
from pyrpoc.services.display_service import DisplayService
from pyrpoc.services.modality_service import ModalityService


class DisplayManagerWidget(QWidget):
    def __init__(
        self,
        display_service: DisplayService,
        modality_service: ModalityService,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.display_service = display_service
        self.modality_service = modality_service

        self._build_ui()
        self._wire_signals()
        self._refresh_available()
        self._refresh_instances()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Display:", self))
        self.display_combo = QComboBox(self)
        add_row.addWidget(self.display_combo, 1)
        self.add_btn = QPushButton("Add", self)
        add_row.addWidget(self.add_btn)
        root.addLayout(add_row)

        root.addWidget(QLabel("Active Displays:", self))
        self.instances_list = QListWidget(self)
        root.addWidget(self.instances_list)

        action_row = QHBoxLayout()
        self.attach_btn = QPushButton("Attach", self)
        self.detach_btn = QPushButton("Detach", self)
        self.remove_btn = QPushButton("Remove", self)
        action_row.addWidget(self.attach_btn)
        action_row.addWidget(self.detach_btn)
        action_row.addWidget(self.remove_btn)
        action_row.addStretch(1)
        root.addLayout(action_row)

        self.status_label = QLabel("Status: ready", self)
        root.addWidget(self.status_label)

        self.display_tabs = QTabWidget(self)
        root.addWidget(self.display_tabs, 1)

    def _wire_signals(self) -> None:
        self.add_btn.clicked.connect(self._on_add_clicked)
        self.attach_btn.clicked.connect(self._on_attach_clicked)
        self.detach_btn.clicked.connect(self._on_detach_clicked)
        self.remove_btn.clicked.connect(self._on_remove_clicked)

        self.display_service.display_added.connect(self._on_display_added)
        self.display_service.display_removed.connect(self._on_display_removed)
        self.display_service.display_error.connect(self._on_display_error)
        self.modality_service.modality_selected.connect(self._on_modality_selected)

    def _refresh_available(self) -> None:
        contract = self.modality_service.get_selected_contract()
        output_type = contract.get("output_data_type")
        allowed_displays = set(contract.get("allowed_displays", []))
        current_key = self._selected_display_key()
        available_rows = {row["key"]: row for row in self.display_service.list_available()}

        self.display_combo.clear()
        if isinstance(output_type, type) and issubclass(output_type, BaseData):
            keys = self.display_service.list_compatible_with(output_type)
        else:
            keys = list(available_rows.keys())

        if allowed_displays:
            keys = [key for key in keys if key in allowed_displays]

        for key in keys:
            row = available_rows.get(key, {"display_name": key})
            self.display_combo.addItem(row.get("display_name", key), key)

        if current_key:
            idx = self.display_combo.findData(current_key)
            if idx >= 0:
                self.display_combo.setCurrentIndex(idx)

    def _refresh_instances(self) -> None:
        current = self._selected_display_id()
        self.instances_list.clear()
        for row in self.display_service.list_instances():
            marker = "attached" if row["attached"] else "detached"
            item = QListWidgetItem(f"{row['name']} [{row['display_id']}] ({marker})")
            item.setData(Qt.ItemDataRole.UserRole, row["display_id"])
            self.instances_list.addItem(item)

        if current:
            for idx in range(self.instances_list.count()):
                item = self.instances_list.item(idx)
                if item.data(Qt.ItemDataRole.UserRole) == current:
                    self.instances_list.setCurrentRow(idx)
                    break

    def _on_add_clicked(self) -> None:
        key = self._selected_display_key()
        if not key:
            self._show_error("No compatible display available for selected modality")
            return

        raw_settings: dict[str, Any] = {}
        display_cls = display_registry.get_class(key)
        if any(display_cls.DISPLAY_PARAMETERS.values()):
            settings = self._prompt_display_parameters(display_cls.DISPLAY_PARAMETERS)
            if settings is None:
                return
            raw_settings = settings

        try:
            display_id, widget = self.display_service.create_display(key, raw_settings)
            self.display_tabs.addTab(widget, display_id)
            self.status_label.setText(f"Status: added {display_id}")
        except Exception as exc:
            self._show_error(str(exc))

    def _on_attach_clicked(self) -> None:
        display_id = self._selected_display_id()
        if not display_id:
            return
        self.display_service.attach(display_id)
        self.status_label.setText(f"Status: attached {display_id}")
        self._refresh_instances()

    def _on_detach_clicked(self) -> None:
        display_id = self._selected_display_id()
        if not display_id:
            return
        self.display_service.detach(display_id)
        self.status_label.setText(f"Status: detached {display_id}")
        self._refresh_instances()

    def _on_remove_clicked(self) -> None:
        display_id = self._selected_display_id()
        if not display_id:
            return
        self.display_service.remove_display(display_id)
        self.status_label.setText(f"Status: removed {display_id}")

    def _on_display_added(self, display_id: str) -> None:
        self.status_label.setText(f"Status: display added {display_id}")
        self._refresh_instances()

    def _on_display_removed(self, display_id: str) -> None:
        for idx in range(self.display_tabs.count()):
            if self.display_tabs.tabText(idx) == display_id:
                widget = self.display_tabs.widget(idx)
                self.display_tabs.removeTab(idx)
                if widget is not None:
                    widget.deleteLater()
                break
        self._refresh_instances()

    def _on_display_error(self, display_id: str, message: str) -> None:
        self.status_label.setText(f"Status: {display_id} error - {message}")

    def _on_modality_selected(self, key: str) -> None:
        del key
        self._refresh_available()

    def _selected_display_id(self) -> str:
        item = self.instances_list.currentItem()
        if item is None:
            return ""
        value = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(value, str):
            return value
        return ""

    def _selected_display_key(self) -> str:
        data = self.display_combo.currentData()
        if isinstance(data, str):
            return data
        return self.display_combo.currentText().strip()

    def _show_error(self, message: str) -> None:
        self.status_label.setText(f"Status: error - {message}")
        QMessageBox.critical(self, "Display Error", message)

    def _prompt_display_parameters(self, parameter_groups: dict[str, list[Parameter]]) -> dict[str, Any] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Display Parameters")
        layout = QVBoxLayout(dialog)

        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        container = QWidget(dialog)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(6, 6, 6, 6)
        container_layout.setSpacing(8)

        widgets: dict[str, QWidget] = {}
        for group_name, parameters in parameter_groups.items():
            group_box = QWidget(container)
            group_layout = QFormLayout(group_box)
            if group_name:
                group_layout.addRow(QLabel(f"[{group_name.capitalize()}]", group_box))
            for param in parameters:
                editor = self._make_editor(param, group_box)
                widgets[param.label] = editor
                group_layout.addRow(param.label, editor)
            container_layout.addWidget(group_box)
        container_layout.addStretch(1)

        scroll.setWidget(container)
        layout.addWidget(scroll)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        values: dict[str, Any] = {}
        for label, widget in widgets.items():
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
            editor.addItems([str(x) for x in param.choices])
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
