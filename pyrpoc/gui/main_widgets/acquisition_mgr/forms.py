from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtWidgets import QFormLayout, QGroupBox, QVBoxLayout, QWidget

from pyrpoc.backend_utils.parameter_utils import BaseParameter
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
    parameter_groups: dict[str, list[BaseParameter]],
    initial_values: dict[str, Any] | None = None,
    on_change: Callable[[], None] | None = None,
) -> None:
    clear_param_form(ui, state)

    if ui.params_layout.count() > 0:
        ui.params_layout.takeAt(ui.params_layout.count() - 1)

    for section_name, parameters in parameter_groups.items():
        section_box = QGroupBox(section_name.capitalize(), ui.params_container)
        form = QFormLayout(section_box)
        for param in parameters:
            widget = param.create_widget(ui.params_container)
            if initial_values and param.label in initial_values:
                param.set_value(widget, initial_values[param.label])
            state.param_widgets[param.label] = (param, widget)
            state.param_defs[param.label] = param
            if on_change is not None:
                param.connect_changed(widget, on_change)
            form.addRow(param.display_label, widget)
        ui.params_layout.addWidget(section_box)

    ui.params_layout.addStretch(1)


def collect_values(widget_map: dict[str, tuple[BaseParameter, QWidget]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for label, (param, widget) in widget_map.items():
        values[label] = param.get_value(widget)
    return values


def apply_values(widget_map: dict[str, tuple[BaseParameter, QWidget]], values: dict[str, Any]) -> None:
    for label, value in values.items():
        item = widget_map.get(label)
        if item is None:
            continue
        param, widget = item
        param.set_value(widget, value)
