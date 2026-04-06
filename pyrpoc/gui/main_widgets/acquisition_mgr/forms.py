from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtWidgets import QFormLayout, QWidget

from pyrpoc.backend_utils.parameter_utils import BaseParameter
from pyrpoc.gui.main_widgets.acquisition_mgr.state import AcquisitionManagerState
from pyrpoc.gui.main_widgets.acquisition_mgr.ui import AcquisitionManagerUI
from pyrpoc.gui.main_widgets.opto_control_mgr.instance_card import InstanceCardWidget


def _build_section_summary(
    params: list[BaseParameter],
    widget_map: dict[str, tuple[BaseParameter, QWidget]],
) -> str:
    parts = []
    for param in params:
        entry = widget_map.get(param.label)
        if entry is None:
            continue
        p, w = entry
        parts.append(f"{p.display_label}: {p.format_summary(w)}")
    return "  |  ".join(parts)


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
        card = InstanceCardWidget(None, section_name.capitalize(), ui.params_container)
        card.set_enable_visible(False)
        card.remove_btn.setVisible(False)

        # Self-contained expand/collapse — no external manager involved
        card.expand_requested.connect(
            lambda _, c=card: c.set_expanded(not c.is_expanded())
        )

        # Build the form inside the card body
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        form_layout.setContentsMargins(4, 4, 4, 4)

        for param in parameters:
            widget = param.create_widget(form_widget)
            if initial_values and param.label in initial_values:
                param.set_value(widget, initial_values[param.label])
            state.param_widgets[param.label] = (param, widget)
            state.param_defs[param.label] = param
            form_layout.addRow(param.display_label, widget)

        card.set_body_widget(form_widget)

        # Set initial summary and wire it to update on any value change
        def _update_summary(c=card, params=parameters) -> None:
            c.set_description(_build_section_summary(params, state.param_widgets))

        _update_summary()

        for param in parameters:
            _, widget = state.param_widgets[param.label]

            def _on_param_change(update=_update_summary, external=on_change) -> None:
                update()
                if external is not None:
                    external()

            param.connect_changed(widget, _on_param_change)

        ui.params_layout.addWidget(card)

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
