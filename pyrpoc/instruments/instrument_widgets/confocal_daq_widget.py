from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.instruments.base_instrument import BaseInstrumentWidget

if False:  # pragma: no cover
    from pyrpoc.instruments.confocal_daq import ConfocalDAQInstrument


class ConfocalDAQInstrumentWidget(BaseInstrumentWidget):
    def __init__(
        self,
        instrument: "ConfocalDAQInstrument",
        on_change: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(instrument, on_change=on_change, parent=parent)
        self.instrument = instrument
        self._channel_leds: list[QToolButton] = []
        self._building_leds = False

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(8)

        grid_box = QGroupBox("Confocal DAQ Configuration", self)
        grid = QGridLayout(grid_box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.device_edit = QLineEdit(self.instrument.device_name, self)
        self.device_edit.editingFinished.connect(self._on_core_params_changed)
        grid.addWidget(QLabel("DAQ Device", self), 0, 0)
        grid.addWidget(self.device_edit, 0, 1)

        self.sample_rate_spin = QDoubleSpinBox(self)
        self.sample_rate_spin.setDecimals(1)
        self.sample_rate_spin.setRange(1.0, 5_000_000.0)
        self.sample_rate_spin.setSingleStep(1_000.0)
        self.sample_rate_spin.setValue(float(self.instrument.sample_rate_hz))
        self.sample_rate_spin.valueChanged.connect(self._on_core_params_changed)
        grid.addWidget(QLabel("Sample Rate (Hz)", self), 1, 0)
        grid.addWidget(self.sample_rate_spin, 1, 1)

        self.fast_ao_spin = QSpinBox(self)
        self.fast_ao_spin.setRange(0, 31)
        self.fast_ao_spin.setValue(int(self.instrument.fast_axis_ao))
        self.fast_ao_spin.valueChanged.connect(self._on_core_params_changed)
        grid.addWidget(QLabel("Fast Axis AO", self), 2, 0)
        grid.addWidget(self.fast_ao_spin, 2, 1)

        self.slow_ao_spin = QSpinBox(self)
        self.slow_ao_spin.setRange(0, 31)
        self.slow_ao_spin.setValue(int(self.instrument.slow_axis_ao))
        self.slow_ao_spin.valueChanged.connect(self._on_core_params_changed)
        grid.addWidget(QLabel("Slow Axis AO", self), 3, 0)
        grid.addWidget(self.slow_ao_spin, 3, 1)

        root.addWidget(grid_box)

        led_box = QGroupBox("Active AI Channels", self)
        led_layout = QVBoxLayout(led_box)
        led_layout.setContentsMargins(8, 8, 8, 8)
        led_layout.setSpacing(6)
        led_layout.addWidget(QLabel("Click to toggle channel activity", self))

        self.led_row = QHBoxLayout()
        self.led_row.setContentsMargins(0, 0, 0, 0)
        self.led_row.setSpacing(6)
        self.led_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        led_layout.addLayout(self.led_row)
        root.addWidget(led_box)

        self._rebuild_led_row()

    def refresh_from_model(self) -> None:
        self.device_edit.setText(self.instrument.device_name)
        self.sample_rate_spin.setValue(float(self.instrument.sample_rate_hz))
        self.fast_ao_spin.setValue(int(self.instrument.fast_axis_ao))
        self.slow_ao_spin.setValue(int(self.instrument.slow_axis_ao))
        self._rebuild_led_row()

    def _on_core_params_changed(self, _value: float | int = 0) -> None:
        del _value
        self.instrument.device_name = self.device_edit.text().strip() or "Dev1"
        self.instrument.sample_rate_hz = float(self.sample_rate_spin.value())
        self.instrument.fast_axis_ao = int(self.fast_ao_spin.value())
        self.instrument.slow_axis_ao = int(self.slow_ao_spin.value())
        self._request_model_persist()

    def _on_led_toggled(self, index: int, checked: bool) -> None:
        if self._building_leds:
            return
        if 0 <= index < len(self.instrument.active_ai_channels):
            self.instrument.active_ai_channels[index] = bool(checked)
            self._request_model_persist()

    def _rebuild_led_row(self) -> None:
        self._building_leds = True
        while self.led_row.count() > 0:
            item = self.led_row.takeAt(0)
            child = item.widget()
            if child is not None:
                child.setParent(None)
                child.deleteLater()
        self._channel_leds.clear()

        for i, ai_idx in enumerate(self.instrument.ai_channel_numbers):
            led = QToolButton(self)
            led.setCheckable(True)
            led.setChecked(bool(self.instrument.active_ai_channels[i]))
            led.setText(f"AI{ai_idx}")
            led.setToolTip(f"Toggle AI{ai_idx}")
            led.setStyleSheet(
                "QToolButton {"
                "padding: 4px 8px;"
                "border: 1px solid palette(mid);"
                "border-radius: 10px;"
                "background: palette(base);"
                "}"
                "QToolButton:checked {"
                "background: palette(highlight);"
                "color: palette(highlighted-text);"
                "border: 1px solid palette(highlight);"
                "font-weight: 700;"
                "}"
            )
            led.toggled.connect(lambda checked, idx=i: self._on_led_toggled(idx, checked))
            self.led_row.addWidget(led)
            self._channel_leds.append(led)

        self._building_leds = False
