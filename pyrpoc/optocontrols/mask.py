from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .base_optocontrol import BaseOptoControl
from .opto_control_registry import opto_control_registry


@opto_control_registry.register("mask")
class MaskOptoControl(BaseOptoControl):
    OPTOCONTROL_KEY = "mask"
    DISPLAY_NAME = "Mask Opto-Control"

    def __init__(self, alias: str | None = None):
        super().__init__(alias=alias)
        self.widget: QWidget | None = None

    def get_widget(self, parent: QWidget | None = None) -> QWidget:
        if self.widget is None:
            host = QWidget(parent)
            layout = QVBoxLayout(host)
            layout.addWidget(QLabel("Mask opto-control UI placeholder", host))
            layout.addWidget(QLabel("Implement this control's full widget later.", host))
            layout.addStretch(1)
            self.widget = host
        elif parent is not None and self.widget.parent() is None:
            self.widget.setParent(parent)
        return self.widget

    def prepare_data_for_acquisition(self) -> dict[str, Any]:
        # maybe it is better to return a data object here instead of a dict
        return {
            "opto_control_key": self.OPTOCONTROL_KEY,
            "alias": self.alias,
        }


Mask = MaskOptoControl
