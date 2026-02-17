from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.backend_utils.contracts import Action, Parameter
from pyrpoc.gui.main_widgets.instrument_mgr.state import InstrumentManagerState
from pyrpoc.gui.main_widgets.instrument_mgr.ui import InstrumentManagerUI


def clear_dynamic_panels(ui: InstrumentManagerUI, state: InstrumentManagerState) -> None:
    build_config_form(ui, state, {})
    build_actions_area(ui, state, [], lambda action_label: None)


def build_config_form(
    ui: InstrumentManagerUI,
    state: InstrumentManagerState,
    parameter_groups: dict[str, list[Parameter]],
) -> None:
    state.config_widgets.clear()
    while ui.config_form.rowCount() > 0:
        ui.config_form.removeRow(0)

    for group_name, parameters in parameter_groups.items():
        if group_name:
            header = QLabel(f"[{group_name.capitalize()}]", ui.config_box)
            header.setStyleSheet("font-weight: bold;")
            ui.config_form.addRow(header)
        for param in parameters:
            editor = make_editor(param, ui.config_box)
            state.config_widgets[param.label] = editor
            ui.config_form.addRow(param.label, editor)


def build_actions_area(
    ui: InstrumentManagerUI,
    state: InstrumentManagerState,
    actions: list[Action],
    run_action: Callable[[str], None],
) -> None:
    state.action_widgets.clear()
    state.actions_by_label.clear()
    state.actions_by_label.update({action.label: action for action in actions})

    while ui.actions_layout.count():
        item = ui.actions_layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()

    for action in actions:
        action_box = QGroupBox(action.label, ui.actions_box)
        box_layout = QVBoxLayout(action_box)

        if action.tooltip:
            action_box.setToolTip(action.tooltip)

        form = QFormLayout()
        action_param_widgets: dict[str, QWidget] = {}
        for param in action.parameters:
            editor = make_editor(param, action_box)
            action_param_widgets[param.label] = editor
            form.addRow(param.label, editor)

        box_layout.addLayout(form)

        run_btn = QPushButton("Run", action_box)
        run_btn.clicked.connect(lambda checked, action_label=action.label: run_action(action_label))
        box_layout.addWidget(run_btn)

        state.action_widgets[action.label] = action_param_widgets
        ui.actions_layout.addWidget(action_box)

    ui.actions_layout.addStretch(1)


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
