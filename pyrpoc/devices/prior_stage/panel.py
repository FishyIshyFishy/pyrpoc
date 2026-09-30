"""The Prior stage's card: connect, read and move, and the motion profile.

A move is sent and then polled on a timer, so the window never blocks on a
travelling stage. The motion profile is written only by its Apply buttons.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

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

from .device import MotionProfile
from .sdk import PriorError

if TYPE_CHECKING:  # pragma: no cover
    from .device import Axis, PriorStage

COORDS = ("x", "y", "z")
AXES: tuple[Axis, ...] = ("xy", "z")
MOTION_HEADINGS = ("Speed (µm/s)", "Acc (µm/s²)", "Jerk (µm/s³)")

# Wide enough for any travel or motion setting, so a value read from the
# controller is never clamped by its box and written back changed.
LIMIT = 1e12
POLL_MS = 100
SPINBOX_WIDTH = 90


def make_spinbox(parent: QWidget, minimum: float, decimals: int) -> QDoubleSpinBox:
    box = QDoubleSpinBox(parent)
    box.setRange(minimum, LIMIT)
    box.setDecimals(decimals)
    # Qt sizes a spinbox to its widest value; with this range that would push
    # the card past the dock's edge, so the layout sets the width instead.
    box.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    box.setMinimumWidth(SPINBOX_WIDTH)
    return box


def make_heading(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    font = label.font()
    font.setBold(True)
    label.setFont(font)
    return label


class PriorStagePanel(QWidget):
    def __init__(self, device: PriorStage, parent: QWidget, on_change: Callable[[], None]) -> None:
        super().__init__(parent)
        self.device = device
        self.on_change = on_change
        self.shown: dict[str, float] = {}
        self.readouts: dict[str, QLabel] = {}
        self.targets: dict[str, QDoubleSpinBox] = {}
        self.motion_boxes: dict[Axis, list[QDoubleSpinBox]] = {}
        self.apply_buttons: dict[Axis, QPushButton] = {}
        self.moving: Axis = "xy"

        self.poll = QTimer(self)
        self.poll.setInterval(POLL_MS)
        self.poll.timeout.connect(self.on_poll)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(self.build_connection_row())
        self.position_section = self.build_position_section()
        layout.addWidget(self.position_section)
        self.motion_section = self.build_motion_section()
        layout.addWidget(self.motion_section)

        self.show_session_status()
        self.update_controls()
        if device.session_open:
            self.read_all()

    def build_connection_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.connect_btn = QPushButton(self)
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        row.addWidget(self.connect_btn)
        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)
        row.addWidget(self.status_label, 1)
        return row

    def build_position_section(self) -> QWidget:
        section = QWidget(self)
        grid = QGridLayout(section)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(make_heading("Position (µm)", section), 0, 0, 1, 3)
        grid.addWidget(QLabel("Current", section), 1, 1)
        grid.addWidget(QLabel("Target", section), 1, 2)
        for row, coord in enumerate(COORDS, start=2):
            grid.addWidget(QLabel(coord.upper(), section), row, 0)
            self.readouts[coord] = QLabel("-", section)
            grid.addWidget(self.readouts[coord], row, 1)
            self.targets[coord] = make_spinbox(section, minimum=-LIMIT, decimals=2)
            grid.addWidget(self.targets[coord], row, 2)
        grid.setColumnStretch(2, 1)
        buttons = QHBoxLayout()
        for text, slot in (
            ("Refresh", self.read_position),
            ("Go XY", self.on_go_xy),
            ("Go Z", self.on_go_z),
            ("Stop", self.on_stop),
        ):
            button = QPushButton(text, section)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        grid.addLayout(buttons, len(COORDS) + 2, 0, 1, 3)
        return section

    def build_motion_section(self) -> QWidget:
        """Axes in columns and quantities in rows, so the table stays two
        inputs wide and fits a narrow dock."""
        section = QWidget(self)
        grid = QGridLayout(section)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(make_heading("Motion", section), 0, 0, 1, len(AXES) + 1)
        for column, axis in enumerate(AXES, start=1):
            grid.addWidget(QLabel(axis.upper(), section), 1, column)
            self.motion_boxes[axis] = []
            for row, _heading in enumerate(MOTION_HEADINGS, start=2):
                box = make_spinbox(section, minimum=0, decimals=3)
                grid.addWidget(box, row, column)
                self.motion_boxes[axis].append(box)
            grid.setColumnStretch(column, 1)
        for row, heading in enumerate(MOTION_HEADINGS, start=2):
            grid.addWidget(QLabel(heading, section), row, 0)
        grid.addLayout(self.build_motion_buttons(section), len(MOTION_HEADINGS) + 2, 0, 1, 3)
        return section

    def build_motion_buttons(self, section: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        read = QPushButton("Read", section)
        read.clicked.connect(self.read_motion)
        row.addWidget(read)
        for axis in AXES:
            apply = QPushButton(f"Apply {axis.upper()}", section)
            # Enabled once this axis has been read, so Apply never writes blanks.
            apply.setEnabled(False)
            self.bind_apply(apply, axis)
            row.addWidget(apply)
            self.apply_buttons[axis] = apply
        return row

    def bind_apply(self, button: QPushButton, axis: Axis) -> None:
        button.clicked.connect(lambda: self.on_apply(axis))

    def attempt(self, action: Callable[[], None]) -> bool:
        """Run one stage call. The hardware is a boundary, so a controller
        fault is shown on the card rather than raised as an unexpected error."""
        try:
            action()
        except PriorError as exc:
            self.status_label.setText(str(exc))
            return False
        return True

    def show_session_status(self) -> None:
        if self.device.session_open:
            self.status_label.setText("Connected")
        else:
            self.status_label.setText(self.device.last_error or "Not connected")

    def update_controls(self) -> None:
        connected = self.device.session_open
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.position_section.setEnabled(connected)
        self.motion_section.setEnabled(connected)
        if not connected:
            for button in self.apply_buttons.values():
                button.setEnabled(False)

    def on_connect_clicked(self) -> None:
        self.poll.stop()
        if self.device.session_open:
            if self.attempt(self.device.close_session):
                self.status_label.setText("Not connected")
        else:
            self.device.try_open_session()
            self.show_session_status()
        self.update_controls()
        if self.device.session_open:
            self.read_all()
        self.on_change()

    def read_all(self) -> None:
        """Everything the controller currently holds; targets start at the
        current position so Go without an edit goes nowhere."""
        self.read_position()
        for coord, value in self.shown.items():
            self.targets[coord].setValue(value)
        self.read_motion()

    def read_position(self) -> None:
        # Separately, so a controller without a Z drive still shows XY.
        self.attempt(self.read_xy)
        self.attempt(self.read_z)

    def read_xy(self) -> None:
        self.shown["x"], self.shown["y"] = self.device.xy_position()
        self.show_readouts()

    def read_z(self) -> None:
        self.shown["z"] = self.device.z_position()
        self.show_readouts()

    def show_readouts(self) -> None:
        for coord, value in self.shown.items():
            self.readouts[coord].setText(f"{value:.2f}")

    def on_go_xy(self) -> None:
        x, y = self.targets["x"].value(), self.targets["y"].value()
        self.start_move("xy", lambda: self.device.move_xy_to(x, y))

    def on_go_z(self) -> None:
        z = self.targets["z"].value()
        self.start_move("z", lambda: self.device.move_z_to(z))

    def on_stop(self) -> None:
        # The poll keeps running, so the readout updates once it has stopped.
        self.attempt(self.device.stop_smoothly)

    def start_move(self, axis: Axis, move: Callable[[], None]) -> None:
        if self.attempt(move):
            self.moving = axis
            self.poll.start()

    def on_poll(self) -> None:
        if not self.attempt(self.check_stopped):
            self.poll.stop()

    def check_stopped(self) -> None:
        if self.device.busy(self.moving):
            return
        self.poll.stop()
        self.read_position()

    def read_motion(self) -> None:
        for axis in AXES:
            self.attempt(lambda a=axis: self.read_axis_motion(a))

    def read_axis_motion(self, axis: Axis) -> None:
        profile = self.device.motion(axis)
        values = (profile.speed, profile.acc, profile.jerk)
        for box, value in zip(self.motion_boxes[axis], values, strict=True):
            box.setValue(value)
        self.apply_buttons[axis].setEnabled(True)

    def on_apply(self, axis: Axis) -> None:
        speed, acc, jerk = (box.value() for box in self.motion_boxes[axis])
        profile = MotionProfile(speed=speed, acc=acc, jerk=jerk)
        self.attempt(lambda: self.device.set_motion(axis, profile))
