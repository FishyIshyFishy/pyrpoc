from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QDoubleSpinBox, QLabel, QVBoxLayout, QWidget

from .base_instrument import BaseInstrument, BaseInstrumentWidget
from .instrument_registry import instrument_registry


@instrument_registry.register("sim_galvo")
class SimGalvoInstrument(BaseInstrument):
    INSTRUMENT_KEY = "sim_galvo"
    DISPLAY_NAME = "Simulated Galvo"

    def __init__(self, alias: str | None = None):
        super().__init__(alias=alias)
        self.x_offset = 0.0
        self.y_offset = 0.0

        self.widget: BaseInstrumentWidget | None = None

    def get_widget(
        self,
        parent: QWidget | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> BaseInstrumentWidget:
        """Called from the expanded instrument card handler.

        The widget is reused and reparented into the card body if the card is reopened.
        """
        if self.widget is None:
            self.widget = SimGalvoInstrumentWidget(self, on_change=on_change, parent=parent)
        elif parent is not None:
            self.widget.setParent(parent)
        if on_change is not None:
            self.widget.on_change = on_change
        return self.widget


class SimGalvoInstrumentWidget(BaseInstrumentWidget):
    """Minimal placeholder controls for the galvo instrument.

    Call flow:
    - created when user expands a card in InstrumentManager
    - spinboxes can trigger on_change to mark dirty state in future persistence paths
    """

    def __init__(
        self,
        instrument: SimGalvoInstrument,
        on_change: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(instrument, on_change=on_change, parent=parent)
        self.instrument = instrument
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Simulated Galvo", self))

        self.x_spin = QDoubleSpinBox(self)
        self.x_spin.setRange(-10.0, 10.0)
        self.x_spin.setValue(float(self.instrument.x_offset))
        self.x_spin.valueChanged.connect(self._on_offsets_changed)

        self.y_spin = QDoubleSpinBox(self)
        self.y_spin.setRange(-10.0, 10.0)
        self.y_spin.setValue(float(self.instrument.y_offset))
        self.y_spin.valueChanged.connect(self._on_offsets_changed)

        layout.addWidget(QLabel("X Offset (V)", self))
        layout.addWidget(self.x_spin)
        layout.addWidget(QLabel("Y Offset (V)", self))
        layout.addWidget(self.y_spin)

    def _on_offsets_changed(self, _value: float) -> None:
        self.instrument.x_offset = float(self.x_spin.value())
        self.instrument.y_offset = float(self.y_spin.value())
        self._request_model_persist()


Galvo = SimGalvoInstrument
