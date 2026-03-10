from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtWidgets import QFormLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from pyrpoc.backend_utils.contracts import Action
from pyrpoc.backend_utils.parameter_utils import BaseParameter
from pyrpoc.gui.main_widgets.instrument_mgr.state import InstrumentManagerState
from pyrpoc.gui.main_widgets.instrument_mgr.ui import InstrumentManagerUI


def build_config_form(
    ui: InstrumentManagerUI,
    state: InstrumentManagerState,
    parameter_groups: dict[str, list[BaseParameter]],
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
            editor = param.create_widget(ui.config_box)
            state.config_widgets[param.label] = editor
            state.config_defs[param.label] = param
            ui.config_form.addRow(param.display_label, editor)


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
        action_param_widgets: dict[str, tuple[BaseParameter, QWidget]] = {}
        for param in action.parameters:
            editor = param.create_widget(action_box)
            action_param_widgets[param.label] = (param, editor)
            form.addRow(param.display_label, editor)

        box_layout.addLayout(form)

        run_btn = QPushButton("Run", action_box)
        run_btn.clicked.connect(lambda checked, action_label=action.label: run_action(action_label))
        box_layout.addWidget(run_btn)

        state.action_widgets[action.label] = action_param_widgets
        ui.actions_layout.addWidget(action_box)

    ui.actions_layout.addStretch(1)


def collect_values(widget_map: dict[str, tuple[BaseParameter, QWidget]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for label, (param, widget) in widget_map.items():
        values[label] = param.get_value(widget)
    return values
