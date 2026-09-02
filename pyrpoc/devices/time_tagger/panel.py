"""Device-specific controls for the TimeTagger, beneath its generated form.

The form itself is generated from ``TaggerConfig`` by ``shell/devices_panel.py``,
so adding a field to the config adds its row with no edit here. What lives in
this file is what a generated form cannot produce: the reachability check.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

if TYPE_CHECKING:  # pragma: no cover
    from .device import TimeTagger


class TimeTaggerPanel(QWidget):
    def __init__(
        self,
        device: "TimeTagger",
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
