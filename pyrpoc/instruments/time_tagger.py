from __future__ import annotations

from .base import BaseInstrument
from .registry import instrument_registry


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
        self.tagger = None

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
        """Create a TimeTagger and store it as self.tagger."""
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
        delays into per-pixel decay curves. Binning happens on the FPGA, so only
        the (n_pixels x n_bins) histogram crosses USB."""
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
