"""Device-specific controls for the CCD, beneath its generated form.

The form itself is generated from ``CcdConfig`` by ``shell/devices_panel.py``,
so adding a field to the config adds its row with no edit here. What lives in
this file is what a generated form cannot produce: connecting, letting go, a
temperature that has to be asked for, and the readout tables that give the
pre-amp gain index its meaning.

Connect and Disconnect are buttons rather than a checkbox, because they are
actions with a physical cost rather than settings -- connecting takes seconds
and starts a cooler that runs for minutes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:  # pragma: no cover
    from .device import CCD

#: How often the temperature line refreshes while the camera is open. The
#: sensor moves over minutes, so this is about the interface not looking dead,
#: not about resolution.
POLL_MS = 2000


class CcdPanel(QWidget):
    def __init__(
        self,
        device: "CCD",
        parent: QWidget | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.device = device
        self.on_change = on_change

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        self.connect_btn = QPushButton("Connect", self)
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        buttons.addWidget(self.connect_btn)
        self.disconnect_btn = QPushButton("Disconnect", self)
        self.disconnect_btn.clicked.connect(self.on_disconnect_clicked)
        buttons.addWidget(self.disconnect_btn)
        self.status_label = QLabel(self)
        buttons.addWidget(self.status_label, 1)
        root.addLayout(buttons)

        self.temp_label = QLabel(self)
        self.temp_label.setEnabled(False)
        root.addWidget(self.temp_label)

        self.tables_label = QLabel(self)
        self.tables_label.setEnabled(False)
        self.tables_label.setWordWrap(True)
        root.addWidget(self.tables_label)

        #: Parented to this widget, so it dies with the card. The devices panel
        #: destroys and rebuilds every card whenever the device list changes,
        #: and a timer that outlived its panel would fire into a deleted widget.
        self.timer = QTimer(self)
        self.timer.setInterval(POLL_MS)
        self.timer.timeout.connect(self.refresh_temperature)

        self.refresh_from_model()

    # -- display --------------------------------------------------------------- #

    def refresh_from_model(self) -> None:
        open_now = self.device.is_open
        self.connect_btn.setEnabled(not open_now)
        self.disconnect_btn.setEnabled(open_now)

        if open_now:
            self.status_label.setText(f"Connected: {self.device.head_model or 'camera'}")
        elif self.device.last_test_ok is None:
            self.status_label.setText("Not connected")
        elif self.device.last_test_ok:
            self.status_label.setText("Not connected (last attempt OK)")
        else:
            self.status_label.setText(self.device.last_error or "FAILED")

        self.refresh_temperature()
        self.refresh_tables()
        if open_now:
            self.timer.start()
        else:
            self.timer.stop()

    def refresh_temperature(self) -> None:
        reading = self.device.temperature()
        if reading is None:
            self.temp_label.setText("Sensor: —")
            return
        celsius, code = reading
        self.temp_label.setText(
            f"Sensor: {celsius:.1f}°C, {self.device.temperature_status(code)}"
        )

    def refresh_tables(self) -> None:
        """What the head offers, so the pre-amp gain index means something.

        An index into a table nobody can see is unusable, and the table can
        only be read from an open camera -- which is why this is here rather
        than in a tooltip on the field.
        """
        if not self.device.is_open:
            self.tables_label.setText("")
            return
        try:
            tables = self.device.speed_tables()
        except Exception as exc:  # noqa: BLE001 - reported in place, not raised
            self.tables_label.setText(f"Could not read the readout tables: {exc}")
            return

        rates = ", ".join(f"{speed:g}" for speed in tables["hs_speeds_mhz"])
        gains = ", ".join(
            f"[{index}] {gain:g}x"
            for index, gain in enumerate(tables["preamp_gains"])
        )
        self.tables_label.setText(
            f"{self.device.width} × {self.device.height} px · "
            f"readout rates (MHz): {rates} · pre-amp gains: {gains}"
        )

    # -- actions ---------------------------------------------------------------- #

    def on_connect_clicked(self) -> None:
        """Open the camera, which blocks for as long as the SDK takes.

        ``repaint`` rather than a worker thread: bringing a Newton up takes
        several seconds and the window is unresponsive for them either way, so
        the honest minimum is to make sure the label saying so is on screen
        before the call rather than after it.
        """
        self.connect_btn.setEnabled(False)
        self.status_label.setText("Connecting… (this takes a few seconds)")
        self.status_label.repaint()

        self.device.test_connection()
        self.refresh_from_model()
        if self.on_change is not None:
            self.on_change()

    def on_disconnect_clicked(self) -> None:
        self.device.close()
        self.refresh_from_model()
        if self.on_change is not None:
            self.on_change()
