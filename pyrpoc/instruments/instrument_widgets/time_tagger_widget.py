from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.instruments.base_instrument import BaseInstrumentWidget

if False:  # pragma: no cover
    from pyrpoc.instruments.time_tagger import TimeTaggerInstrument


class TimeTaggerInstrumentWidget(BaseInstrumentWidget):
    def __init__(
        self,
        instrument: "TimeTaggerInstrument",
        on_change: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(instrument, on_change=on_change, parent=parent)
        self.instrument = instrument

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(8)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.test_btn = QPushButton("Test Connection", self)
        self.test_btn.clicked.connect(self._on_test_clicked)
        row.addWidget(self.test_btn)

        self.status_label = QLabel(self)
        row.addWidget(self.status_label)
        row.addStretch()

        root.addLayout(row)

        self._sync_status_label()

    def refresh_from_model(self) -> None:
        self._sync_status_label()

    def _on_test_clicked(self) -> None:
        self.test_btn.setEnabled(False)
        self.status_label.setText("Testing\u2026")
        ok = self.instrument.test_connection()
        self.status_label.setText("OK" if ok else "FAILED")
        self.test_btn.setEnabled(True)
        self._request_model_persist()

    def _sync_status_label(self) -> None:
        val = self.instrument.last_test_ok
        if val is None:
            self.status_label.setText("Not tested")
        elif val:
            self.status_label.setText("OK")
        else:
            self.status_label.setText("FAILED")
