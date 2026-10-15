"""The ToupTek camera's card: connect and disconnect, so a camera ToupView held
at launch can be picked up once ToupView is closed."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from .sdk import ToupcamError

if TYPE_CHECKING:  # pragma: no cover
    from .device import ToupcamCamera


class ToupcamPanel(QWidget):
    def __init__(
        self, device: ToupcamCamera, parent: QWidget, on_change: Callable[[], None]
    ) -> None:
        super().__init__(parent)
        self.device = device
        self.on_change = on_change

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.connect_btn = QPushButton(self)
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        row.addWidget(self.connect_btn)
        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)
        row.addWidget(self.status_label, 1)

        self.show_session_status()

    def show_session_status(self) -> None:
        session = self.device.session
        if session is not None:
            self.status_label.setText(f"Connected: {session.camera.name}")
        else:
            self.status_label.setText(self.device.last_error or "Not connected")
        self.connect_btn.setText("Disconnect" if session is not None else "Connect")

    def on_connect_clicked(self) -> None:
        if self.device.session_open:
            # The hardware is a boundary: a fault is shown on the card.
            try:
                self.device.close_session()
            except ToupcamError as exc:
                self.device.last_error = str(exc)
        else:
            self.device.try_open_session()
        self.show_session_status()
        self.on_change()
