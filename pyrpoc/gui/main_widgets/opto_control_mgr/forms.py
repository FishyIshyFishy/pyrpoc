from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QWidget

from pyrpoc.backend_utils.parameter_utils import BaseParameter


def collect_values(widget_map: dict[str, tuple[BaseParameter, QWidget]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for label, (param, widget) in widget_map.items():
        values[label] = param.get_value(widget)
    return values
