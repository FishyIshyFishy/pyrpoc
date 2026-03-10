from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.backend_utils.parameter_utils import BaseParameter


def collect_values(widget_map: dict[str, tuple[BaseParameter, QWidget]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for label, (param, widget) in widget_map.items():
        values[label] = param.get_value(widget)
    return values


def prompt_display_parameters(
    owner: QWidget,
    parameter_groups: dict[str, list[BaseParameter]],
) -> dict[str, Any] | None:
    dialog = QDialog(owner)
    dialog.setWindowTitle("Display Parameters")
    layout = QVBoxLayout(dialog)

    scroll = QScrollArea(dialog)
    scroll.setWidgetResizable(True)
    container = QWidget(dialog)
    container_layout = QVBoxLayout(container)
    container_layout.setContentsMargins(6, 6, 6, 6)
    container_layout.setSpacing(8)

    widgets: dict[str, tuple[BaseParameter, QWidget]] = {}
    for group_name, parameters in parameter_groups.items():
        group_box = QWidget(container)
        group_layout = QFormLayout(group_box)
        if group_name:
            group_layout.addRow(QLabel(f"[{group_name.capitalize()}]", group_box))
        for param in parameters:
            editor = param.create_widget(group_box)
            widgets[param.label] = (param, editor)
            group_layout.addRow(param.display_label, editor)
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

    return collect_values(widgets)
