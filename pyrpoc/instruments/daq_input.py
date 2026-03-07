from __future__ import annotations

from PyQt6.QtWidgets import QLineEdit, QLabel, QVBoxLayout, QWidget
from collections.abc import Callable

from .base_instrument import BaseInstrument, BaseInstrumentWidget
from .instrument_registry import instrument_registry


@instrument_registry.register("sim_daq_input")
class SimDAQInput(BaseInstrument):
    INSTRUMENT_KEY = "sim_daq_input"
    DISPLAY_NAME = "Simulated DAQ Input"

    def __init__(self, alias: str | None = None):
        super().__init__(alias=alias)
        self.widget: BaseInstrumentWidget | None = None
        self.device_name = "Dev1"

    def get_widget(
        self,
        parent: QWidget | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> BaseInstrumentWidget:
        """Called when the user expands the instrument card in InstrumentManager.

        This is the same call path as BaseOptoControl: widget construction is local
        to this instrument so modality/service code never imports UI details.
        """
        if self.widget is None:
            self.widget = SimDAQInputWidget(self, on_change=on_change, parent=parent)
        elif parent is not None:
            self.widget.setParent(parent)
        if on_change is not None:
            self.widget.on_change = on_change
        return self.widget


class SimDAQInputWidget(BaseInstrumentWidget):
    """Minimal placeholder DAQ widget used by the simplified API.

    Call flow:
    - created by `SimDAQInput.get_widget` while a card is expanded
    - edits can call `on_change` if persistence ever needs dirty-state tracking later
    """

    def __init__(
        self,
        instrument: SimDAQInput,
        on_change: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(instrument, on_change=on_change, parent=parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Simulated DAQ Input", self))
        device = QLineEdit(getattr(instrument, "device_name", "Dev1"), self)
        device.setReadOnly(True)
        layout.addWidget(device)
        self.device_name = device


DAQInput = SimDAQInput
