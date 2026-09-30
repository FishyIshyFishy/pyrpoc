"""The Swabian TimeTagger: its SDK handle, its wiring, and the Flim measurement.

Channel numbers and trigger voltages are configuration rather than run
parameters because they describe how the tagger is cabled and thresholded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from pyrpoc.structs import params as P
from pyrpoc.structs.device import Device, DeviceError, device_registry

if TYPE_CHECKING:  # pragma: no cover
    from PyQt6.QtWidgets import QWidget


class TaggerError(DeviceError):
    """A TimeTagger operation failed."""


class FlimMeasurement(Protocol):
    """The part of the SDK's ``Flim`` this code calls, so programs can type it
    without importing the SDK."""

    def stop(self) -> None: ...

    def getCurrentFrameEx(self) -> Any: ...


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

    def check_reachable(self) -> bool:
        self.create_tagger()
        self.free_tagger()
        return True

    def create_tagger(self) -> None:
        from Swabian import TimeTagger as sdk

        self.tagger = sdk.createTimeTagger()

    def free_tagger(self) -> None:
        if self.tagger is None:
            return
        from Swabian import TimeTagger as sdk

        sdk.freeTimeTagger(self.tagger)
        self.tagger = None

    def configure_for_flim(self) -> None:
        """Set per-channel trigger levels, and the laser delay that slides the
        decay curve into the histogram window."""
        if self.tagger is None:
            raise TaggerError("create_tagger() must be called before configure_for_flim()")
        c = self.config
        self.tagger.setTriggerLevel(c.laser_channel, c.laser_trigger_v)
        self.tagger.setTriggerLevel(c.detector_channel, c.detector_trigger_v)
        self.tagger.setTriggerLevel(c.pixel_channel, c.pixel_trigger_v)
        self.tagger.setTriggerLevel(c.frame_channel, c.frame_trigger_v)
        if c.laser_input_delay_ps:
            self.tagger.setInputDelay(c.laser_channel, int(c.laser_input_delay_ps))

    def start_flim_measurement(
        self, *, n_pixels: int, n_bins: int, binwidth_ps: int
    ) -> FlimMeasurement:
        """Start the hardware Flim measurement. Binning happens on the FPGA and
        only the (n_pixels x n_bins) histogram crosses USB, so the laser stream
        cannot overflow the link the way raw time-tag streaming does."""
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

    def stop_flim_measurement(self, flim: FlimMeasurement) -> None:
        """Stop the measurement and free the tagger, once per run."""
        flim.stop()
        self.free_tagger()

    def panel(self, parent: QWidget, on_change: Callable[[], None]) -> QWidget:
        from .panel import TimeTaggerPanel

        return TimeTaggerPanel(self, parent, on_change)
