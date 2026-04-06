from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

        box = QGroupBox("TimeTagger Configuration", self)
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.serial_edit = QLineEdit(self.instrument.serial, self)
        self.serial_edit.setPlaceholderText("blank = first found device")
        self.serial_edit.editingFinished.connect(self._on_serial_changed)
        grid.addWidget(QLabel("Serial", self), 0, 0)
        grid.addWidget(self.serial_edit, 0, 1)

        root.addWidget(box)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)

        self.connect_btn = QPushButton("Connect", self)
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        btn_row.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect", self)
        self.disconnect_btn.clicked.connect(self._on_disconnect_clicked)
        btn_row.addWidget(self.disconnect_btn)

        self.status_label = QLabel(self)
        btn_row.addWidget(self.status_label)
        btn_row.addStretch()

        root.addLayout(btn_row)

        self._sync_ui_to_state()

    def refresh_from_model(self) -> None:
        self.serial_edit.setText(self.instrument.serial)
        self._sync_ui_to_state()

    def _on_serial_changed(self) -> None:
        self.instrument.serial = self.serial_edit.text().strip()
        self._request_model_persist()

    def _on_connect_clicked(self) -> None:
        self.instrument.serial = self.serial_edit.text().strip()
        self.instrument.connected = True
        self._sync_ui_to_state()
        self._request_model_persist()

    def _on_disconnect_clicked(self) -> None:
        self.instrument.connected = False
        self._sync_ui_to_state()
        self._request_model_persist()

    def _sync_ui_to_state(self) -> None:
        connected = self.instrument.connected
        self.serial_edit.setReadOnly(connected)
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        self.status_label.setText("Connected" if connected else "Disconnected")
