"""The devices panel: what is configured, and how it is wired.

Each card's body is a form generated from the device's config plus whatever
controls the device supplies, so a new config field needs no edit here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QScrollArea, QVBoxLayout, QWidget

from pyrpoc.src.structs.device import Device
from pyrpoc.src.structs.panel import Panel
from pyrpoc.src.structs.params import FieldContext
from pyrpoc.src.structs.registries import device_registry

from ..components.cards import RemovableCardWidget
from ..components.param_form import ParamForm

if TYPE_CHECKING:  # pragma: no cover
    from pyrpoc.src.app.application import Application


class DevicesPanel(Panel):
    display_name = "Devices"

    def __init__(self, app: Application):
        super().__init__()
        self.app = app

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        top = QHBoxLayout()
        self.type_combo = QComboBox(self)
        for key in device_registry.keys():
            self.type_combo.addItem(device_registry.get(key).display_name, key)
        add_btn = QPushButton("Add", self)
        add_btn.clicked.connect(lambda: self.app.add_device(self.type_combo.currentData()))
        top.addWidget(self.type_combo, 1)
        top.addWidget(add_btn)
        root.addLayout(top)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        self.content = QWidget(scroll_area)
        self.instances_layout = QVBoxLayout(self.content)
        self.instances_layout.setContentsMargins(0, 0, 0, 0)
        self.instances_layout.setSpacing(6)
        scroll_area.setWidget(self.content)
        root.addWidget(scroll_area, 1)

        self.app.devices_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        while (item := self.instances_layout.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        for device in self.app.devices:
            self.instances_layout.addWidget(self.build_card(device))
        self.instances_layout.addStretch(1)

    def build_card(self, device: Device) -> RemovableCardWidget:
        card = RemovableCardWidget(device.name, self.content)
        card.set_description(device.summary())
        card.remove_requested.connect(lambda d=device: self.app.remove_device(d))

        body = QWidget(card)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        form = ParamForm([device.config], body, cards=False, context=FieldContext())
        form.changed.connect(lambda d=device, c=card: self.on_config_changed(d, c))
        layout.addWidget(form)
        extra = device.panel(body, lambda d=device, c=card: self.on_config_changed(d, c))
        if extra is not None:
            layout.addWidget(extra)

        card.set_body_widget(body)
        return card

    def on_config_changed(self, device: Device, card: RemovableCardWidget) -> None:
        card.set_description(device.summary())
        card.title_label.setText(device.name)
        self.app.state_changed.emit()
