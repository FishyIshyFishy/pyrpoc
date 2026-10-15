"""The Andor CCD's card: connect, and watch and steer the cooling.

The target temperature is edited in the config form above this card; Set
Temperature sends it to the camera.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .sdk import AndorError

if TYPE_CHECKING:  # pragma: no cover
    from .device import AndorCCD

POLL_MS = 1000


class AndorCcdPanel(QWidget):
    def __init__(self, device: AndorCCD, parent: QWidget, on_change: Callable[[], None]) -> None:
        super().__init__(parent)
        self.device = device
        self.on_change = on_change

        self.poll = QTimer(self)
        self.poll.setInterval(POLL_MS)
        self.poll.timeout.connect(self.on_poll)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.build_connection_row())
        self.camera_label = QLabel(self)
        layout.addWidget(self.camera_label)
        self.cooling_section = self.build_cooling_section()
        layout.addWidget(self.cooling_section)

        self.show_session_status()
        self.update_controls()

    def build_connection_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.connect_btn = QPushButton(self)
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        row.addWidget(self.connect_btn)
        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)
        row.addWidget(self.status_label, 1)
        return row

    def build_cooling_section(self) -> QWidget:
        section = QWidget(self)
        layout = QHBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        self.temperature_label = QLabel("-", section)
        layout.addWidget(self.temperature_label, 1)
        for text, slot in (
            ("Set Temperature", lambda: self.attempt(self.device.apply_target_temperature)),
            ("Cooler On", lambda: self.attempt(lambda: self.device.set_cooler(True))),
            ("Cooler Off", lambda: self.attempt(lambda: self.device.set_cooler(False))),
        ):
            button = QPushButton(text, section)
            button.clicked.connect(slot)
            layout.addWidget(button)
        return section

    def attempt(self, action: Callable[[], None]) -> bool:
        """Run one camera call. The hardware is a boundary, so a fault is
        shown on the card rather than raised as an unexpected error."""
        try:
            action()
        except AndorError as exc:
            self.status_label.setText(str(exc))
            return False
        return True

    def show_session_status(self) -> None:
        session = self.device.session
        if session is None:
            self.status_label.setText(self.device.last_error or "Not connected")
            self.camera_label.setText("")
            return
        info = session.info
        self.status_label.setText("Connected")
        self.camera_label.setText(
            f"{info.head_model} #{info.serial}, "
            f"{info.width}x{info.height} px of {info.pixel_um:g} µm"
        )

    def update_controls(self) -> None:
        connected = self.device.session_open
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.cooling_section.setEnabled(connected)
        if connected:
            self.poll.start()
            self.on_poll()
        else:
            self.poll.stop()
            self.temperature_label.setText("-")

    def on_connect_clicked(self) -> None:
        if self.device.session_open:
            self.attempt(self.device.close_session)
        else:
            self.device.try_open_session()
        self.show_session_status()
        self.update_controls()
        self.on_change()

    def on_poll(self) -> None:
        if not self.attempt(self.show_temperature):
            self.poll.stop()

    def show_temperature(self) -> None:
        reading = self.device.temperature()
        if reading.celsius is None:
            self.temperature_label.setText(reading.status)
        else:
            self.temperature_label.setText(f"{reading.celsius:.1f} °C, {reading.status}")
