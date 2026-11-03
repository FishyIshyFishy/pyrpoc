"""Editing the configuration of a device that has no panel of its own."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget

from pyrpoc.core.device import Device
from pyrpoc.core.program_inputs import FieldContext
from pyrpoc.shared.widgets.param_form import ConfigValues, ParamForm, config_form_section

from .inventory import DeviceInventory


class ConfigureDialog(QDialog):
    """A device's configuration form, applying each edit as it is made."""

    # So there is nothing to cancel and the only button is Close.

    def __init__(self, device: Device, inventory: DeviceInventory, parent: QWidget):
        super().__init__(parent)
        # Shown modeless, so nothing else holds it once it is closed.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Configure {device.name}")
        layout = QVBoxLayout(self)
        form = ParamForm(
            [config_form_section(device.config_cls)],
            ConfigValues(device),
            self,
            cards=False,
            context=FieldContext(),
        )
        form.changed.connect(inventory.device_changed)
        layout.addWidget(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
