from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.backend_utils.contracts import Parameter
from pyrpoc.services.modality_service import ModalityService


class AcquisitionManagerWidget(QWidget):
    def __init__(self, modality_service: ModalityService, parent: QWidget | None = None):
        super().__init__(parent)
        self.modality_service = modality_service
        self.param_widgets: dict[str, QWidget] = {}
        self.param_defs: dict[str, Parameter] = {}
        self._continuous_timer = QTimer(self)
        self._continuous_timer.setInterval(250)
        self._continuous_timer.timeout.connect(self._acquire_single_frame)

        self._build_ui()
        self._wire_signals()
        self._populate_modalities()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(QLabel("Modality:", self))
        self.modality_combo = QComboBox(self)
        top.addWidget(self.modality_combo, 1)
        self.refresh_btn = QPushButton("Refresh", self)
        top.addWidget(self.refresh_btn)
        root.addLayout(top)

        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start (One Frame)", self)
        self.continuous_btn = QPushButton("Continuous", self)
        self.stop_btn = QPushButton("Stop", self)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.continuous_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch(1)
        root.addLayout(controls)

        self.status_label = QLabel("Status: idle", self)
        root.addWidget(self.status_label)

        self.params_container = QWidget(self)
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.setSpacing(8)
        self.params_layout.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.params_container)
        root.addWidget(scroll, 1)

    def _wire_signals(self) -> None:
        self.refresh_btn.clicked.connect(self._populate_modalities)
        self.modality_combo.currentTextChanged.connect(self._on_modality_selected)
        self.start_btn.clicked.connect(self._on_start_clicked)
        self.continuous_btn.clicked.connect(self._on_continuous_clicked)
        self.stop_btn.clicked.connect(self._on_stop_clicked)

        self.modality_service.modality_selected.connect(self._handle_modality_selected)
        self.modality_service.requirements_changed.connect(self._handle_requirements_changed)
        self.modality_service.acq_started.connect(lambda: self.status_label.setText("Status: acquiring"))
        self.modality_service.acq_stopped.connect(lambda: self.status_label.setText("Status: stopped"))
        self.modality_service.acq_error.connect(self._on_service_error)

    def _populate_modalities(self) -> None:
        current = self.modality_combo.currentText()
        self.modality_combo.blockSignals(True)
        self.modality_combo.clear()
        rows = self.modality_service.list_available()
        for row in rows:
            self.modality_combo.addItem(row["key"])
        self.modality_combo.blockSignals(False)

        if current and self.modality_combo.findText(current) >= 0:
            self.modality_combo.setCurrentText(current)
        elif self.modality_combo.count() > 0:
            self.modality_combo.setCurrentIndex(0)
            self._on_modality_selected(self.modality_combo.currentText())

    def _on_modality_selected(self, key: str) -> None:
        if not key:
            return
        try:
            self.modality_service.select_modality(key)
        except Exception as exc:
            self._handle_error(str(exc))

    def _handle_modality_selected(self, key: str) -> None:
        del key
        parameter_groups = self.modality_service.get_selected_parameters()
        self._build_param_form(parameter_groups)

    def _build_param_form(self, parameter_groups: dict[str, list[Parameter]]) -> None:
        self.param_widgets.clear()
        self.param_defs.clear()
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for section_name, parameters in parameter_groups.items():
            section_box = QGroupBox(section_name.capitalize(), self.params_container)
            form = QFormLayout(section_box)
            for param in parameters:
                widget = self._make_editor(param)
                self.param_widgets[param.label] = widget
                self.param_defs[param.label] = param
                form.addRow(param.label, widget)
            self.params_layout.addWidget(section_box)

        self.params_layout.addStretch(1)

    def _make_editor(self, param: Parameter) -> QWidget:
        default = param.default

        if param.param_type is int:
            editor = QSpinBox(self.params_container)
            editor.setMinimum(int(param.minimum if param.minimum is not None else -1_000_000))
            editor.setMaximum(int(param.maximum if param.maximum is not None else 1_000_000))
            editor.setValue(int(default if default is not None else 0))
            if param.tooltip:
                editor.setToolTip(param.tooltip)
            return editor
        if param.param_type is float:
            editor = QDoubleSpinBox(self.params_container)
            editor.setDecimals(6)
            editor.setMinimum(float(param.minimum if param.minimum is not None else -1e12))
            editor.setMaximum(float(param.maximum if param.maximum is not None else 1e12))
            editor.setSingleStep(float(param.step if param.step is not None else 0.1))
            editor.setValue(float(default if default is not None else 0.0))
            if param.tooltip:
                editor.setToolTip(param.tooltip)
            return editor
        if param.param_type is bool:
            editor = QCheckBox(self.params_container)
            editor.setChecked(bool(default))
            if param.tooltip:
                editor.setToolTip(param.tooltip)
            return editor
        if param.param_type is str and param.choices:
            editor = QComboBox(self.params_container)
            editor.addItems([str(choice) for choice in param.choices])
            if default is not None:
                editor.setCurrentText(str(default))
            if param.tooltip:
                editor.setToolTip(param.tooltip)
            return editor

        editor = QLineEdit(self.params_container)
        editor.setText("" if default is None else str(default))
        if param.param_type is Path:
            editor.setPlaceholderText("Path")
        if param.tooltip:
            editor.setToolTip(param.tooltip)
        return editor

    def _read_params(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for label, widget in self.param_widgets.items():
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

    def _configure_modality(self) -> None:
        params = self._read_params()
        self.modality_service.configure(params)

    def _on_start_clicked(self) -> None:
        try:
            self._configure_modality()
            self.modality_service.start()
            self._acquire_single_frame()
            self.modality_service.stop()
        except Exception as exc:
            self._handle_error(str(exc))

    def _on_continuous_clicked(self) -> None:
        try:
            self._configure_modality()
            self.modality_service.start()
            self._continuous_timer.start()
            self.status_label.setText("Status: continuous acquisition")
        except Exception as exc:
            self._handle_error(str(exc))

    def _acquire_single_frame(self) -> None:
        try:
            self.modality_service.acquire_once()
        except Exception as exc:
            self._continuous_timer.stop()
            try:
                self.modality_service.stop()
            except Exception as stop_exc:
                self.status_label.setText(f"Status: stop error - {stop_exc}")
            self._handle_error(str(exc))

    def _on_stop_clicked(self) -> None:
        self._continuous_timer.stop()
        try:
            self.modality_service.stop()
        except Exception as exc:
            self._handle_error(str(exc))

    def _handle_requirements_changed(self, ok: bool, missing_names: list[str]) -> None:
        if ok:
            self.status_label.setText("Status: ready")
            return
        text = ", ".join(missing_names)
        self.status_label.setText(f"Status: missing instruments -> {text}")

    def _handle_error(self, message: str) -> None:
        self.status_label.setText(f"Status: error - {message}")
        QMessageBox.critical(self, "Acquisition Error", message)

    def _on_service_error(self, message: str) -> None:
        self.status_label.setText(f"Status: error - {message}")
