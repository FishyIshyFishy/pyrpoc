from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QWidget

from pyrpoc.instruments.base_instrument import BaseInstrument, BaseInstrumentWidget
from pyrpoc.instruments.instrument_registry import instrument_registry
from pyrpoc.instruments.instrument_widgets.time_tagger_widget import TimeTaggerInstrumentWidget


@instrument_registry.register("time_tagger")
class TimeTaggerInstrument(BaseInstrument):
    instrument_key = "time_tagger"
    display_name = "Swabian TimeTagger"

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
        self.tagger = None

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
        frame_ch: int,
        laser_trigger_v: float,
        detector_trigger_v: float,
        pixel_trigger_v: float,
        frame_trigger_v: float,
        *,
        laser_input_delay_ps: int = 0,
    ) -> None:
        """Set per-channel trigger levels and the laser input delay used to
        slide the decay curve into the histogram window."""
        if self.tagger is None:
            raise RuntimeError("create_tagger() must be called before configure_for_flim()")
        self.tagger.setTriggerLevel(laser_ch, laser_trigger_v)
        self.tagger.setTriggerLevel(detector_ch, detector_trigger_v)
        self.tagger.setTriggerLevel(pixel_ch, pixel_trigger_v)
        self.tagger.setTriggerLevel(frame_ch, frame_trigger_v)
        if laser_input_delay_ps:
            self.tagger.setInputDelay(laser_ch, int(laser_input_delay_ps))

    def create_flim_measurement(
        self,
        laser_ch: int,
        detector_ch: int,
        pixel_ch: int,
        frame_ch: int,
        n_pixels: int,
        n_bins: int,
        binwidth_ps: int,
    ) -> object:
        """Create the hardware Flim measurement that histograms laser-to-photon
        delays into per-pixel decay curves.

        Binning happens on the FPGA and only the (n_pixels x n_bins) histogram
        crosses USB, so the 80 MHz laser stream is never transferred and the
        device cannot overflow the way raw TimeTagStream acquisition does. The
        pixel/frame channels carry the DAQ scan markers that assign photons to
        pixels.
        """
        if self.tagger is None:
            raise RuntimeError("create_tagger() must be called before create_flim_measurement()")
        from Swabian import TimeTagger
        return TimeTagger.Flim(
            self.tagger,
            start_channel=laser_ch,
            click_channel=detector_ch,
            pixel_begin_channel=pixel_ch,
            n_pixels=int(n_pixels),
            n_bins=int(n_bins),
            binwidth=int(binwidth_ps),
            frame_begin_channel=frame_ch,
            n_frame_average=1,
        )
