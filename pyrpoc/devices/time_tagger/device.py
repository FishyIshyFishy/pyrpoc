"""The Swabian TimeTagger: its SDK handle, its wiring, and the Flim measurement.

Driver moved from ``instruments/time_tagger.py``; its panel moved from
``instruments/instrument_widgets/time_tagger_widget.py``. They changed together
constantly and lived in different folders, which is the case section 6.1 is
built on.

Channel numbers and trigger voltages move out of the FLIM parameter form and
into this device's configuration: they describe how the tagger is cabled and
thresholded, which is calibration, not a per-run choice. Laser frequency,
histogram bins and bin width stay run parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyrpoc.core import params as P
from pyrpoc.core.errors import TaggerError

from ..base import Device
from ..registry import device_registry

if TYPE_CHECKING:  # pragma: no cover
    from PyQt6.QtWidgets import QWidget


@dataclass
class TaggerConfig(P.Group):
    laser_channel: int = P.int_field(
        "Laser Channel", 1, minimum=1, tooltip="Input channel for the laser sync (start)"
    )
    detector_channel: int = P.int_field(
        "Detector Channel", 2, minimum=1, tooltip="Input channel for the SPAD detector (click)"
    )
    pixel_channel: int = P.int_field(
        "Pixel Channel", 3, minimum=1, tooltip="Input channel for the DAQ pixel clock"
    )
    frame_channel: int = P.int_field(
        "Frame Channel", 4, minimum=1, tooltip="Input channel for the DAQ frame-start trigger"
    )
    laser_trigger_v: float = P.float_field(
        "Laser Trigger V", 0.05, tooltip="Trigger threshold for the laser sync channel (V)"
    )
    detector_trigger_v: float = P.float_field(
        "Detector Trigger V", 0.2, tooltip="Trigger threshold for the detector channel (V)"
    )
    pixel_trigger_v: float = P.float_field(
        "Pixel Trigger V", 0.5, tooltip="Trigger threshold for the pixel clock channel (V)"
    )
    frame_trigger_v: float = P.float_field(
        "Frame Trigger V", 0.5, tooltip="Trigger threshold for the frame trigger channel (V)"
    )
    laser_input_delay_ps: int = P.int_field(
        "Laser Input Delay (ps)",
        0,
        tooltip="Delay added to the laser channel to position the decay in the histogram window",
    )


@device_registry.register("time_tagger")
class TimeTagger(Device):
    display_name = "Swabian TimeTagger"
    owns_connection = True
    config_cls = TaggerConfig

    config: TaggerConfig

    def __init__(self, instance_id: str | None = None, user_label: str | None = None):
        super().__init__(instance_id=instance_id, user_label=user_label)
        self.tagger = None

    def summary(self) -> str:
        if self.last_test_ok is None:
            return "Connection: not tested"
        return "Connection: OK" if self.last_test_ok else "Connection: FAILED"

    # -- connection -------------------------------------------------------- #

    def check_reachable(self) -> bool:
        self.create_tagger()
        self.free_tagger()
        return True

    def create_tagger(self) -> None:
        """Create a TimeTagger and store it as self.tagger."""
        from Swabian import TimeTagger as sdk

        self.tagger = sdk.createTimeTagger()

    def free_tagger(self) -> None:
        """Free self.tagger and clear the reference."""
        if self.tagger is not None:
            try:
                from Swabian import TimeTagger as sdk

                sdk.freeTimeTagger(self.tagger)
            except Exception:
                pass
            self.tagger = None

    # -- FLIM -------------------------------------------------------------- #

    def configure_for_flim(self) -> None:
        """Set per-channel trigger levels and the laser input delay used to
        slide the decay curve into the histogram window."""
        if self.tagger is None:
            raise TaggerError("create_tagger() must be called before configure_for_flim()")
        c = self.config
        self.tagger.setTriggerLevel(c.laser_channel, c.laser_trigger_v)
        self.tagger.setTriggerLevel(c.detector_channel, c.detector_trigger_v)
        self.tagger.setTriggerLevel(c.pixel_channel, c.pixel_trigger_v)
        self.tagger.setTriggerLevel(c.frame_channel, c.frame_trigger_v)
        if c.laser_input_delay_ps:
            self.tagger.setInputDelay(c.laser_channel, int(c.laser_input_delay_ps))

    def start_flim_measurement(self, *, n_pixels: int, n_bins: int, binwidth_ps: int) -> object:
        """Create the hardware Flim measurement that histograms laser-to-photon
        delays into per-pixel decay curves.

        Binning happens on the FPGA and only the (n_pixels x n_bins) histogram
        crosses USB, so the 80 MHz laser stream is never transferred and the
        device cannot overflow the way raw TimeTagStream acquisition does. The
        pixel/frame channels carry the DAQ scan markers that assign photons to
        pixels.
        """
        if self.tagger is None:
            raise TaggerError("create_tagger() must be called before start_flim_measurement()")
        from Swabian import TimeTagger as sdk

        c = self.config
        return sdk.Flim(
            self.tagger,
            start_channel=c.laser_channel,
            click_channel=c.detector_channel,
            pixel_begin_channel=c.pixel_channel,
            n_pixels=int(n_pixels),
            n_bins=int(n_bins),
            binwidth=int(binwidth_ps),
            frame_begin_channel=c.frame_channel,
            n_frame_average=1,
        )

    def stop_flim_measurement(self, flim: object | None) -> None:
        """Stop the measurement and free the tagger. Called once per run, from
        the program's ``finally`` -- not once per frame as v3.0 did."""
        if flim is not None:
            try:
                flim.stop()  # type: ignore[attr-defined]
            except Exception:
                pass
        self.free_tagger()

    # -- panel ------------------------------------------------------------- #

    def panel(self, parent: "QWidget | None" = None, on_change=None) -> "QWidget | None":
        from .panel import TimeTaggerPanel

        return TimeTaggerPanel(self, parent=parent, on_change=on_change)
