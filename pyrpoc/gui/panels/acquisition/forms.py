from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtWidgets import QFormLayout, QWidget

from pyrpoc.gui.panels.acquisition.state import AcquisitionManagerState
from pyrpoc.gui.panels.acquisition.ui import AcquisitionManagerUI
from pyrpoc.gui.panels.cards import BaseCardWidget
from pyrpoc.gui.widgets import parameter_widgets as pw
from pyrpoc.structs.parameters import BaseParameter


def build_section_summary(
    params: list[BaseParameter],
    widget_map: dict[str, tuple[BaseParameter, QWidget]],
) -> str:
    parts = []
    for param in params:
        entry = widget_map.get(param.label)
        if entry is None:
            continue
        p, w = entry
        parts.append(f"{p.display_label}: {pw.format_summary(p, w)}")
    return "  |  ".join(parts)


def clear_param_form(ui: AcquisitionManagerUI, state: AcquisitionManagerState) -> None:
    state.param_widgets.clear()
    state.param_defs.clear()
    while (item := ui.params_layout.takeAt(0)) is not None:
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
        card = BaseCardWidget(None, section_name.capitalize(), ui.params_container)
        card.set_toggle_visible(False)
        card.expand_requested.connect(
            lambda _, c=card: c.set_expanded(not c.is_expanded())
        )

        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        form_layout.setContentsMargins(4, 4, 4, 4)

        for param in parameters:
            widget = pw.create_widget(param, form_widget)
            if initial_values and param.label in initial_values:
                pw.set_value(param, widget, initial_values[param.label])
            state.param_widgets[param.label] = (param, widget)
            state.param_defs[param.label] = param
            form_layout.addRow(param.display_label, widget)

        card.set_body_widget(form_widget)

        def update_summary(c=card, params=parameters) -> None:
            c.set_description(build_section_summary(params, state.param_widgets))

        update_summary()

        for param in parameters:
            _, widget = state.param_widgets[param.label]

            def on_param_change(update=update_summary, external=on_change) -> None:
                update()
                if external is not None:
                    external()

            pw.connect_changed(param, widget, on_param_change)

        ui.params_layout.addWidget(card)

    ui.params_layout.addStretch(1)


def collect_values(widget_map: dict[str, tuple[BaseParameter, QWidget]]) -> dict[str, Any]:
    return {label: pw.get_value(param, widget) for label, (param, widget) in widget_map.items()}


def apply_values(widget_map: dict[str, tuple[BaseParameter, QWidget]], values: dict[str, Any]) -> None:
    for label, value in values.items():
        item = widget_map.get(label)
        if item is None:
            continue
        param, widget = item
        pw.set_value(param, widget, value)
