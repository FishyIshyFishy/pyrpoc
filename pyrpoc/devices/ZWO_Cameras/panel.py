"""Device-specific controls for the ZWO camera, beneath its generated form.

The form itself is generated from ``ZWOCameraConfig`` by ``shell/devices_panel.py``.
What lives here is what a generated form cannot produce: the reachability
check, and a temperature readout that only means something once the camera has
been opened by a run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

if TYPE_CHECKING:  # pragma: no cover
    from .device import ZWOCamera


class ZWOCameraPanel(QWidget):
    def __init__(
        self,
        device: "ZWOCamera",
        parent: QWidget | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.device = device
        self.on_change = on_change

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.test_btn = QPushButton("Test Connection", self)
        self.test_btn.clicked.connect(self.on_test_clicked)
        row.addWidget(self.test_btn)
        self.status_label = QLabel(self)
        row.addWidget(self.status_label)
        self.temp_btn = QPushButton("Poll Temperature", self)
        self.temp_btn.clicked.connect(self.on_temp_clicked)
        row.addWidget(self.temp_btn)
        self.temp_label = QLabel(self)
        row.addWidget(self.temp_label)
        row.addStretch(1)

        self.refresh_from_model()

    def refresh_from_model(self) -> None:
        ok = self.device.last_test_ok
        if ok is None:
            self.status_label.setText("Not tested")
        elif ok:
            self.status_label.setText("OK")
        else:
            self.status_label.setText(self.device.last_error or "FAILED")

    def on_test_clicked(self) -> None:
        self.test_btn.setEnabled(False)
        self.status_label.setText("Testing…")
        self.device.test_connection()
        self.refresh_from_model()
        self.test_btn.setEnabled(True)
        if self.on_change is not None:
            self.on_change()

    def on_temp_clicked(self) -> None:
        reading = self.device.poll_temperature()
        if reading is None:
            self.temp_label.setText("Camera not open (run must be started)")
            return
        temp_c, power_pct = reading
        self.temp_label.setText(f"{temp_c:.1f}°C, cooler {power_pct}%")
