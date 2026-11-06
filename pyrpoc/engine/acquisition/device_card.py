"""``DeviceCard``: the Acquisition panel's card choosing which device each name a program
chooses a device under stands for."""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from pyrpoc.core.device import Device
from pyrpoc.utils.widgets.cards import BaseCardWidget

from .runs import Runs


class DeviceCard(BaseCardWidget):
    """One dropdown per name chosen, listing the inventory's devices of its class."""

    def __init__(
        self, runs: Runs, program_key: str, chooses: dict[str, type[Device]], parent: QWidget
    ):
        super().__init__("Devices", parent)
        self.runs = runs
        self.program_key = program_key
        self.chooses = chooses
        self.combos: dict[str, QComboBox] = {}
        # What each row of each dropdown stands for.
        self.options: dict[str, list[Device]] = {}
        body = QWidget(self)
        column = QVBoxLayout(body)
        column.setContentsMargins(4, 4, 4, 4)
        column.setSpacing(2)
        for name in self.chooses:
            combo = QComboBox(body)
            combo.setToolTip(f"Which device this program uses as its {name}.")
            combo.activated.connect(lambda index, n=name: self.on_picked(n, index))
            self.combos[name] = combo
            # Above the dropdown, as input cards show their labels, so a narrow panel never cuts it.
            column.addWidget(QLabel(name, body))
            column.addWidget(combo)
        self.set_body_widget(body)
        runs.devices_chosen.connect(self.reload)
        runs.inventory.changed.connect(self.reload)
        self.reload()

    def reload(self) -> None:
        """Show the devices the inventory holds now, and which is chosen for each name."""
        chosen = self.runs.chosen_devices(self.program_key)
        inventory = self.runs.inventory
        for name, device_type in self.chooses.items():
            combo = self.combos[name]
            options = [each for each in inventory.devices if isinstance(each, device_type)]
            combo.blockSignals(True)
            combo.clear()
            for each in options:
                combo.addItem(inventory.name(each))
            # Blank when there is no device of the class, or the program's other names hold them.
            combo.setPlaceholderText(f"None free: add another {device_type.display_name}")
            combo.setCurrentIndex(options.index(chosen[name]) if name in chosen else -1)
            combo.setEnabled(bool(options))
            combo.blockSignals(False)
            self.options[name] = options
        self.set_summary(
            [
                (name, inventory.name(chosen[name]) if name in chosen else "none")
                for name in self.chooses
            ]
        )

    def on_picked(self, name: str, index: int) -> None:
        self.runs.choose_device(name, self.options[name][index])

    def detach(self) -> None:
        """Stop following the runs and devices, before this card is thrown away."""
        self.runs.devices_chosen.disconnect(self.reload)
        self.runs.inventory.changed.disconnect(self.reload)
