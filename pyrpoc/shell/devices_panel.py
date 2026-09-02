"""The devices panel: what is configured, and how it is wired.

Replaces gui/main_widgets/instrument_mgr/. Each card's body is a form generated
from the device's own config plus whatever extra controls the device supplies
from its panel.py -- so adding a config field adds its row with no edit here.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.devices.base import Device
from pyrpoc.devices.registry import device_registry

from .app import Application
from .cards import RemovableCardWidget
from .param_form import ParamForm


class DevicesPanel(QWidget):
    def __init__(self, app: Application, parent: QWidget | None = None):
        super().__init__(parent)
        self.app = app
        self.cards: dict[Device, RemovableCardWidget] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        top = QHBoxLayout()
        self.type_combo = QComboBox(self)
        for key in device_registry.keys():
            self.type_combo.addItem(device_registry.get(key).display_name, key)
        self.add_btn = QPushButton("Add", self)
        top.addWidget(self.type_combo, 1)
        top.addWidget(self.add_btn)
        root.addLayout(top)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.content = QWidget(self.scroll)
        self.instances_layout = QVBoxLayout(self.content)
        self.instances_layout.setContentsMargins(0, 0, 0, 0)
        self.instances_layout.setSpacing(6)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll, 1)

        self.add_btn.clicked.connect(self.on_add_clicked)
        self.app.devices_changed.connect(self.refresh)
        self.refresh()

    def on_add_clicked(self) -> None:
        key = self.type_combo.currentData()
        if isinstance(key, str):
            self.app.add_device(key)

    def refresh(self) -> None:
        while self.instances_layout.count():
            item = self.instances_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.cards.clear()

        for device in self.app.devices:
            card = self.build_card(device)
            self.cards[device] = card
            self.instances_layout.addWidget(card)
        self.instances_layout.addStretch(1)

    def build_card(self, device: Device) -> RemovableCardWidget:
        card = RemovableCardWidget(device, device.name, self.content)
        card.set_toggle_visible(False)
        card.set_description(device.summary())
        card.expand_requested.connect(lambda _obj, c=card: c.set_expanded(not c.is_expanded()))
        card.remove_requested.connect(lambda obj: self.app.remove_device(obj))

        body = QWidget(card)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        if device.config is not None:
            form = ParamForm(device.config, body, cards=False)
            form.changed.connect(lambda d=device, c=card: self.on_config_changed(d, c))
            layout.addWidget(form)

        extra = device.panel(parent=body, on_change=lambda d=device, c=card: self.on_config_changed(d, c))
        if extra is not None:
            layout.addWidget(extra)

        card.set_body_widget(body)
        return card

    def on_config_changed(self, device: Device, card: RemovableCardWidget) -> None:
        card.set_description(device.summary())
        card.title_label.setText(device.name)
        self.app.state_changed.emit()
