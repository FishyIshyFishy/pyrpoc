"""The Simulated Dish's panel: its settings, and where its stage is and moves to."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core.panels import DevicePanel
from pyrpoc.utils.widgets.param_form import ConfigValues, ParamForm, config_form_section

from .dish import SimulatedDish

COORDS = ("x", "y", "z")
TRAVEL_UM = 1e6
POLL_MS = 200


class SimulatedDishPanel(DevicePanel):
    def __init__(self, device: SimulatedDish, on_change: Callable[[], None]) -> None:
        super().__init__(device, on_change)
        self.device = device
        self.readouts: dict[str, QLabel] = {}
        self.targets: dict[str, QDoubleSpinBox] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        form = ParamForm(
            [config_form_section(device.config_cls)], ConfigValues(device), self, cards=False
        )
        form.changed.connect(on_change)
        layout.addWidget(form)
        layout.addWidget(self.build_position_section())
        self.time_label = QLabel(self)
        layout.addWidget(self.time_label)
        layout.addStretch(1)

        self.poll = QTimer(self)
        self.poll.setInterval(POLL_MS)
        self.poll.timeout.connect(self.show_position)
        self.poll.start()
        self.show_position()

    def build_position_section(self) -> QWidget:
        section = QWidget(self)
        grid = QGridLayout(section)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QLabel("Current", section), 0, 1)
        grid.addWidget(QLabel("Target", section), 0, 2)
        for row, coord in enumerate(COORDS, start=1):
            grid.addWidget(QLabel(f"{coord.upper()} (µm)", section), row, 0)
            self.readouts[coord] = QLabel("-", section)
            grid.addWidget(self.readouts[coord], row, 1)
            box = QDoubleSpinBox(section)
            box.setRange(-TRAVEL_UM, TRAVEL_UM)
            box.setDecimals(2)
            box.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            box.setMinimumWidth(90)
            self.targets[coord] = box
            grid.addWidget(box, row, 2)
        grid.setColumnStretch(2, 1)
        buttons = QHBoxLayout()
        for text, slot in (
            ("Go XY", self.on_go_xy),
            ("Go Z", self.on_go_z),
            ("Stop", self.on_stop),
        ):
            button = QPushButton(text, section)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        grid.addLayout(buttons, len(COORDS) + 1, 0, 1, 3)
        return section

    def show_position(self) -> None:
        x, y = self.device.xy_position()
        position = {"x": x, "y": y, "z": self.device.z_position()}
        for coord, value in position.items():
            self.readouts[coord].setText(f"{value:.2f}")
        minutes = self.device.now() / 60.0
        self.time_label.setText(f"Dish time {minutes:.1f} min")

    def on_go_xy(self) -> None:
        self.device.move_xy_to(self.targets["x"].value(), self.targets["y"].value())

    def on_go_z(self) -> None:
        self.device.move_z_to(self.targets["z"].value())

    def on_stop(self) -> None:
        self.device.stop_smoothly()
