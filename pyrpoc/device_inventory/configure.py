"""Editing the configuration of a device that has no panel of its own."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget

from pyrpoc.qt_components.param_form import ParamForm
from pyrpoc.structs.plugins.devices import Device
from pyrpoc.structs.plugins.params import FieldContext

from .inventory import DeviceInventory


class ConfigureDialog(QDialog):
    """Edits apply as they are made, like a panel's form, so there is nothing
    to cancel and the only button is Close."""

    def __init__(self, device: Device, inventory: DeviceInventory, parent: QWidget):
        super().__init__(parent)
        # Shown modeless, so nothing else holds it once it is closed.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Configure {device.name}")
        layout = QVBoxLayout(self)
        form = ParamForm([device.config], self, cards=False, context=FieldContext())
        form.changed.connect(inventory.device_changed)
        layout.addWidget(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
