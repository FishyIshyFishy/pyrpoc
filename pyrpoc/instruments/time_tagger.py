from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QWidget

from pyrpoc.instruments.base_instrument import BaseInstrument, BaseInstrumentWidget
from pyrpoc.instruments.instrument_registry import instrument_registry
from pyrpoc.instruments.instrument_widgets.time_tagger_widget import TimeTaggerInstrumentWidget


@instrument_registry.register("time_tagger")
class TimeTaggerInstrument(BaseInstrument):
    INSTRUMENT_KEY = "time_tagger"
    DISPLAY_NAME = "Swabian TimeTagger"

    def __init__(
        self,
        alias: str | None = None,
        *,
        instance_id: str | None = None,
        user_label: str | None = None,
        connected: bool = False,
    ):
        super().__init__(
            alias=alias,
            instance_id=instance_id,
            user_label=user_label,
            connected=connected,
        )
        self.last_test_ok: bool | None = None
        self.widget: BaseInstrumentWidget | None = None

    def get_widget(
        self,
        parent: QWidget | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> BaseInstrumentWidget:
        if self.widget is None:
            self.widget = TimeTaggerInstrumentWidget(self, on_change=on_change, parent=parent)
        elif parent is not None:
            self.widget.setParent(parent)
        if on_change is not None:
            self.widget.on_change = on_change
        return self.widget

    def get_collapsed_summary(self) -> str:
        if self.last_test_ok is None:
            status = "not tested"
        elif self.last_test_ok:
            status = "OK"
        else:
            status = "FAILED"
        return f"Connection: {status}"

    def test_connection(self) -> bool:
        try:
            self.create_tagger()
            self.free_tagger()
            self.last_test_ok = True
        except Exception:
            self.last_test_ok = False
        return self.last_test_ok

    def create_tagger(self) -> None:
        """Create a TimeTagger from self.serial and store it as self.tagger."""
        from Swabian import TimeTagger
        self.tagger = TimeTagger.createTimeTagger()

    def free_tagger(self) -> None:
        """Free self.tagger and clear the reference."""
        if self.tagger is not None:
            try:
                from Swabian import TimeTagger
                TimeTagger.freeTimeTagger(self.tagger)
            except Exception:
                pass
            self.tagger = None

    def configure_for_flim(
        self,
        laser_ch: int,
        detector_ch: int,
        pixel_ch: int,
        laser_trigger_v: float,
        detector_trigger_v: float,
        pixel_trigger_v: float,
        laser_event_divider: int = 1,
    ) -> None:
        """Set trigger levels and optional event divider on self.tagger."""
        self.tagger.setTriggerLevel(laser_ch, laser_trigger_v)
        self.tagger.setTriggerLevel(detector_ch, detector_trigger_v)
        self.tagger.setTriggerLevel(pixel_ch, pixel_trigger_v)
        if int(laser_event_divider) > 1:
            self.tagger.setEventDivider(laser_ch, int(laser_event_divider))

    def create_flim_stream(
        self,
        laser_ch: int,
        detector_ch: int,
        pixel_ch: int,
        buffer_size: int = 4_000_000,
    ) -> object:
        """Create and return a TimeTagStream on self.tagger for FLIM acquisition."""
        from Swabian import TimeTagger
        return TimeTagger.TimeTagStream(
            tagger=self.tagger,
            n_max_events=buffer_size,
            channels=[laser_ch, detector_ch, pixel_ch],  # pyright:ignore
        )
