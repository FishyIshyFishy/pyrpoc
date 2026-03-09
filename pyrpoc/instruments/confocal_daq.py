from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QWidget

from pyrpoc.instruments.base_instrument import BaseInstrument, BaseInstrumentWidget
from pyrpoc.instruments.instrument_registry import instrument_registry
from pyrpoc.instruments.instrument_widgets import ConfocalDAQInstrumentWidget


@instrument_registry.register("confocal_daq")
class ConfocalDAQInstrument(BaseInstrument):
    INSTRUMENT_KEY = "confocal_daq"
    DISPLAY_NAME = "Confocal DAQ"

    def __init__(self, alias: str | None = None):
        super().__init__(alias=alias)
        self.device_name: str = "Dev1"
        self.sample_rate_hz: float = 100_000.0
        self.ai_channel_numbers: list[int] = list(range(9))
        self.active_ai_channels: list[bool] = [True] * 9
        self.fast_axis_ao: int = 0
        self.slow_axis_ao: int = 1

        self.widget: BaseInstrumentWidget | None = None

    def get_widget(
        self,
        parent: QWidget | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> BaseInstrumentWidget:
        if self.widget is None:
            self.widget = ConfocalDAQInstrumentWidget(self, on_change=on_change, parent=parent)
        elif parent is not None:
            self.widget.setParent(parent)
        if on_change is not None:
            self.widget.on_change = on_change
        return self.widget

    def prepare_for_acquisition(self) -> tuple[object, ...]:
        active_ai = [
            ai_idx
            for ai_idx, enabled in zip(self.ai_channel_numbers, self.active_ai_channels, strict=False)
            if enabled
        ]
        return (
            self.device_name,
            float(self.sample_rate_hz),
            len(self.ai_channel_numbers),
            list(self.ai_channel_numbers),
            active_ai,
            int(self.fast_axis_ao),
            int(self.slow_axis_ao),
        )

    def get_collapsed_summary(self) -> str:
        active_ai = [
            ai_idx
            for ai_idx, enabled in zip(self.ai_channel_numbers, self.active_ai_channels, strict=False)
            if enabled
        ]
        ai_text = ", ".join(str(ai) for ai in active_ai) if active_ai else "none"
        return f"AIs: {ai_text}; Galvos: {int(self.fast_axis_ao)},{int(self.slow_axis_ao)}"


ConfocalDAQ = ConfocalDAQInstrument
