from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.backend_utils.contracts import Parameter
from pyrpoc.gui.main_widgets.acquisition_mgr.state import AcquisitionManagerState
from pyrpoc.gui.main_widgets.acquisition_mgr.ui import AcquisitionManagerUI


def clear_param_form(ui: AcquisitionManagerUI, state: AcquisitionManagerState) -> None:
    state.param_widgets.clear()
    state.param_defs.clear()
    while ui.params_layout.count():
        item = ui.params_layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
    ui.params_layout.addStretch(1)


def build_param_form(
    ui: AcquisitionManagerUI,
    state: AcquisitionManagerState,
    parameter_groups: dict[str, list[Parameter]],
) -> None:
    clear_param_form(ui, state)

    if ui.params_layout.count() > 0:
        ui.params_layout.takeAt(ui.params_layout.count() - 1)

    for section_name, parameters in parameter_groups.items():
        section_box = QGroupBox(section_name.capitalize(), ui.params_container)
        form = QFormLayout(section_box)
        for param in parameters:
            widget = make_editor(param, ui.params_container)
            state.param_widgets[param.label] = widget
            state.param_defs[param.label] = param
            form.addRow(param.label, widget)
        ui.params_layout.addWidget(section_box)

    ui.params_layout.addStretch(1)


def make_editor(param: Parameter, parent: QWidget) -> QWidget:
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
        editor.addItems([str(choice) for choice in param.choices])
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


def collect_values(widget_map: dict[str, QWidget]) -> dict[str, Any]:
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
