from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QWidget

from pyrpoc.gui.widgets import parameter_widgets as pw
from pyrpoc.structs.parameters import BaseParameter


def collect_values(widget_map: dict[str, tuple[BaseParameter, QWidget]]) -> dict[str, Any]:
    return {label: pw.get_value(param, widget) for label, (param, widget) in widget_map.items()}
