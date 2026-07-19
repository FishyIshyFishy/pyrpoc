"""Swabian TimeTagger instrument for FLIM. Ported from main, Qt-free.

Runs real hardware only. The Swabian library is imported lazily so this module
loads on machines without it; calls fail with a clear error when no tagger is present.
"""

from __future__ import annotations

from pyrpoc_next.instruments.base import Instrument
from pyrpoc_next.instruments.registry import instrument_registry
from pyrpoc_next.structs.keys import InstrumentKey
from pyrpoc_next.structs.status import ConnectionStatus


@instrument_registry.register
class TimeTagger(Instrument):
    """Photon time-tagger that builds per-pixel decay histograms for lifetime imaging."""

    key = InstrumentKey.time_tagger
    display_name = "Swabian TimeTagger"

    def __init__(self):
        super().__init__()
        self.tagger = None

    def test_connection(self) -> bool:
        try:
            self.create_tagger()
            self.free_tagger()
            self.status = ConnectionStatus.ok
            return True
        except Exception:
            self.status = ConnectionStatus.failed
            return False

    def create_tagger(self) -> None:
        """Open the hardware tagger."""
        from Swabian import TimeTagger as swabian

        self.tagger = swabian.createTimeTagger()

    def free_tagger(self) -> None:
        """Release the hardware tagger."""
        if self.tagger is None:
            return
        from Swabian import TimeTagger as swabian

        swabian.freeTimeTagger(self.tagger)
        self.tagger = None

    def configure_for_flim(self, laser_ch: int, detector_ch: int, pixel_ch: int, frame_ch: int,
                           laser_trigger_v: float, detector_trigger_v: float, pixel_trigger_v: float,
                           frame_trigger_v: float, laser_input_delay_ps: int = 0) -> None:
        """Set per-channel trigger levels and the laser input delay."""
        if self.tagger is None:
            raise RuntimeError("create_tagger() must run before configure_for_flim()")
        self.tagger.setTriggerLevel(laser_ch, laser_trigger_v)
        self.tagger.setTriggerLevel(detector_ch, detector_trigger_v)
        self.tagger.setTriggerLevel(pixel_ch, pixel_trigger_v)
        self.tagger.setTriggerLevel(frame_ch, frame_trigger_v)
        if laser_input_delay_ps:
            self.tagger.setInputDelay(laser_ch, int(laser_input_delay_ps))

    def create_flim_measurement(self, laser_ch: int, detector_ch: int, pixel_ch: int, frame_ch: int,
                                n_pixels: int, n_bins: int, binwidth_ps: int) -> object:
        """Create the on-FPGA Flim measurement that histograms photon delays per pixel."""
        if self.tagger is None:
            raise RuntimeError("create_tagger() must run before create_flim_measurement()")
        from Swabian import TimeTagger as swabian

        return swabian.Flim(
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
