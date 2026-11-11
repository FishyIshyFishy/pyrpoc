"""The Zaber stage's panel: its port and axis, connect, and where the stage is: read it, home it,
send it to a position or step it."""

# A move is sent and then polled on a timer, so the window never blocks on a travelling stage.

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

from pyrpoc.utils.widgets.device_panel import DeviceLink, DevicePanel, device_panel_registry
from pyrpoc.utils.widgets.param_form import ConfigValues, ParamForm, config_form_section

from .device import ZaberError

if TYPE_CHECKING:  # pragma: no cover
    from .device import ZaberStage

# Wide enough for any travel, so a position read from the stage is never clamped by its box.
LIMIT = 1e9
POLL_MS = 100
SPINBOX_WIDTH = 100
STEP_UM = 10.0


def make_spinbox(parent: QWidget, value: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox(parent)
    box.setRange(-LIMIT, LIMIT)
    box.setDecimals(3)
    box.setValue(value)
    # Qt sizes a spinbox to its widest value; with this range that would push the panel past
    # the dock's edge, so the layout sets the width instead.
    box.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    box.setMinimumWidth(SPINBOX_WIDTH)
    return box


def make_heading(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    font = label.font()
    font.setBold(True)
    label.setFont(font)
    return label


@device_panel_registry.register("zaber_stage")
class ZaberStagePanel(DevicePanel):
    def __init__(self, device: ZaberStage, link: DeviceLink) -> None:
        super().__init__(device, link)
        self.device = device
        # What the controls last showed, so a sync only re-reads the stage when the connection
        # actually changed.
        self.connected = device.session_open

        self.poll = QTimer(self)
        self.poll.setInterval(POLL_MS)
        self.poll.timeout.connect(self.on_poll)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        form = ParamForm(
            [config_form_section(device.config_cls)],
            ConfigValues(device),
            self,
            cards=False,
        )
        form.changed.connect(link.changed)
        layout.addWidget(form)
        layout.addLayout(self.build_connection_row())
        self.position_section = self.build_position_section()
        layout.addWidget(self.position_section)
        layout.addStretch(1)

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
        grid.addWidget(make_heading("Position (µm)", section), 0, 0, 1, 4)
        grid.addWidget(QLabel("Current", section), 1, 0)
        self.readout = QLabel("-", section)
        grid.addWidget(self.readout, 1, 1)
        self.homed_label = QLabel("", section)
        grid.addWidget(self.homed_label, 1, 2, 1, 2)
        grid.addWidget(QLabel("Target", section), 2, 0)
        self.target = make_spinbox(section, 0.0)
        grid.addWidget(self.target, 2, 1, 1, 2)
        grid.addWidget(self.button("Go", self.on_go, section), 2, 3)
        grid.addWidget(QLabel("Step", section), 3, 0)
        self.step = make_spinbox(section, STEP_UM)
        self.step.setToolTip("How far − and + move the stage")
        grid.addWidget(self.step, 3, 1)
        grid.addWidget(self.button("−", lambda: self.on_step(-1.0), section), 3, 2)
        grid.addWidget(self.button("+", lambda: self.on_step(1.0), section), 3, 3)
        grid.setColumnStretch(1, 1)
        buttons = QHBoxLayout()
        for text, slot in (
            ("Refresh", self.read_position),
            ("Home", self.on_home),
            ("Stop", self.on_stop),
        ):
            buttons.addWidget(self.button(text, slot, section))
        grid.addLayout(buttons, 4, 0, 1, 4)
        return section

    def button(self, text: str, slot: Callable[[], None], parent: QWidget) -> QPushButton:
        button = QPushButton(text, parent)
        button.clicked.connect(slot)
        return button

    def attempt(self, action: Callable[[], None]) -> bool:
        """Run one stage call, showing a fault on the panel instead of raising it."""
        # The hardware is a boundary.
        try:
            action()
        except ZaberError as exc:
            self.status_label.setText(str(exc))
            return False
        return True

    def show_session_status(self) -> None:
        if self.device.session_open:
            self.status_label.setText("Connected")
        else:
            self.status_label.setText(self.link.fault() or "Not connected")

    def update_controls(self) -> None:
        connected = self.device.session_open
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.position_section.setEnabled(connected)

    def on_connect_clicked(self) -> None:
        """Connect or disconnect; ``sync`` then updates the controls, as it does when the
        inventory connects the stage."""
        if self.device.session_open:
            self.poll.stop()
            self.attempt(self.device.close_session)
        else:
            # Shown here too, since a failed connect leaves nothing for sync to change.
            self.link.connect()
            self.show_session_status()
        self.link.changed()

    def sync(self) -> None:
        if self.device.session_open == self.connected:
            return
        self.connected = self.device.session_open
        self.poll.stop()
        self.show_session_status()
        self.update_controls()
        if self.connected:
            self.read_all()

    def read_all(self) -> None:
        """Where the stage is; the target starts there, so Go without an edit goes nowhere."""
        if self.attempt(self.read_now):
            self.target.setValue(self.device.position_um())

    def read_position(self) -> None:
        self.attempt(self.read_now)

    def read_now(self) -> None:
        self.readout.setText(f"{self.device.position_um():.3f}")
        # Absolute moves count from home, so a stage not homed says so.
        self.homed_label.setText("homed" if self.device.homed() else "not homed: press Home")

    def on_go(self) -> None:
        target = self.target.value()
        self.start_move(lambda: self.device.move_to(target))

    def on_step(self, sign: float) -> None:
        step = sign * self.step.value()
        self.start_move(lambda: self.device.move_by(step))

    def on_home(self) -> None:
        self.start_move(self.device.home)

    def on_stop(self) -> None:
        # The poll keeps running, so the readout updates once it has stopped.
        self.attempt(self.device.stop)

    def start_move(self, move: Callable[[], None]) -> None:
        if self.attempt(move):
            self.poll.start()

    def on_poll(self) -> None:
        if not self.attempt(self.follow_move):
            self.poll.stop()

    def follow_move(self) -> None:
        """Show where the stage is now, and stop polling once it has stopped."""
        self.read_now()
        if not self.device.busy():
            self.poll.stop()
