"""The ZWO ASI camera: SDK handle, ROI/gain/temperature control, video-capture streaming.

Ported from a standalone PyQt widefield app where the camera worker, ROI/bit
depth/gain control and the video-capture loop all lived inline in a QThread.
Camera identity (which USB camera) and the cooling setpoint that protects the
sensor between runs are calibration and stay on this device; frame geometry,
bit depth, exposure and gain change run to run and live on ``WidefieldGroup``
in ``programs/widefield.py`` -- the same split ``daq/device.py`` draws between
wiring and ``ScanGroup``.

The camera handle is opened and closed once per run, inside
``programs/widefield.py``, rather than held open across runs: the same
lifecycle ``TimeTagger.create_tagger``/``free_tagger`` uses around a FLIM run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyrpoc.core import params as P
from pyrpoc.core.errors import ZWOCameraError

from ..base import Device
from ..registry import device_registry

if TYPE_CHECKING:  # pragma: no cover
    from PyQt6.QtWidgets import QWidget


#: Bundled alongside this driver so the app runs with no separate SDK install.
_SDK_DLL_PATH = os.path.join(os.path.dirname(__file__), "ASICamera2.dll")

_sdk_ready = False


def _load_sdk():
    """Import ``zwoasi`` and load the native ASI library exactly once."""
    global _sdk_ready
    import zwoasi as asi

    if not _sdk_ready:
        try:
            asi.init(_SDK_DLL_PATH)
        except Exception as exc:
            raise ZWOCameraError(f"could not load ASICamera2.dll: {exc}") from exc
        _sdk_ready = True
    return asi


@dataclass
class ZWOCameraConfig(P.Group):
    camera_index: int = P.int_field(
        "Camera Index", 0, minimum=0, tooltip="Which ASI camera to open, 0 for the first USB device"
    )
    target_temp_c: int = P.int_field(
        "Cooler Target (C)",
        -10,
        minimum=-40,
        maximum=20,
        tooltip="Thermoelectric cooler setpoint, applied while the camera is open",
    )


#: Sensor-brightness band an auto-gain step nudges the manual gain towards, one
#: pair per bit depth. Carried over from the source app's preview loop.
_AUTO_GAIN_BAND = {8: (70, 140), 16: (18000, 35000)}


@device_registry.register("zwo_camera")
class ZWOCamera(Device):
    display_name = "ZWO ASI Camera"
    owns_connection = True
    config_cls = ZWOCameraConfig

    config: ZWOCameraConfig

    def __init__(self, instance_id: str | None = None, user_label: str | None = None):
        super().__init__(instance_id=instance_id, user_label=user_label)
        self.camera = None
        self.width = 0
        self.height = 0
        self.bit_depth = 8
        self._capturing = False

    def summary(self) -> str:
        if self.last_test_ok is None:
            return "Connection: not tested"
        return "Connection: OK" if self.last_test_ok else "Connection: FAILED"

    # -- connection ---------------------------------------------------------- #

    def check_reachable(self) -> bool:
        asi = _load_sdk()
        camera = asi.Camera(self.config.camera_index)
        try:
            camera.get_camera_property()
        finally:
            camera.close()
        return True

    def open(self) -> None:
        """Open the physical camera and start its cooler towards the setpoint."""
        if self.camera is not None:
            self.close()
        asi = _load_sdk()
        try:
            self.camera = asi.Camera(self.config.camera_index)
        except Exception as exc:
            raise ZWOCameraError(f"could not open camera {self.config.camera_index}: {exc}") from exc
        self.camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 100)
        try:
            self.camera.set_control_value(asi.ASI_COOLER_ON, 1)
            self.camera.set_control_value(asi.ASI_TARGET_TEMP, int(self.config.target_temp_c))
        except Exception:
            pass  # not every ASI model is a cooled model

    def close(self) -> None:
        if self.camera is None:
            return
        if self._capturing:
            self.stop_capture()
        try:
            self.camera.set_control_value(_load_sdk().ASI_COOLER_ON, 0)
        except Exception:
            pass
        try:
            self.camera.close()
        except Exception:
            pass
        self.camera = None

    # -- geometry, exposure, gain -------------------------------------------- #

    def configure(
        self, *, width: int, height: int, binning: int, bit_depth: int, exposure_s: float, gain: int
    ) -> None:
        """Apply ROI, bit depth, exposure and gain to the open camera.

        Width/height are rounded down to the ROI alignment the SDK enforces --
        32px for 8-bit, 8px for 16-bit -- and the crop is centered on the
        sensor, matching the source app's ``apply_roi_settings``.
        """
        if self.camera is None:
            raise ZWOCameraError("open() must be called before configure()")
        asi = _load_sdk()
        was_capturing = self._capturing
        if was_capturing:
            self.stop_capture()

        props = self.camera.get_camera_property()
        align = 8 if bit_depth == 16 else 32
        self.camera.set_image_type(asi.ASI_IMG_RAW16 if bit_depth == 16 else asi.ASI_IMG_RAW8)
        width = max(align, (int(width) // align) * align)
        height = max(align, (int(height) // align) * align)

        start_x = max(0, (props["MaxWidth"] // binning - width) // 2)
        start_y = max(0, (props["MaxHeight"] // binning - height) // 2)
        self.camera.set_roi(start_x=start_x, start_y=start_y, width=width, height=height, bins=binning)
        self.camera.set_control_value(asi.ASI_EXPOSURE, int(exposure_s * 1_000_000))
        self.camera.set_control_value(asi.ASI_GAIN, int(gain))

        self.width = width
        self.height = height
        self.bit_depth = bit_depth

        if was_capturing:
            self.start_capture()

    # -- streaming ------------------------------------------------------------ #

    def start_capture(self) -> None:
        if self.camera is None:
            raise ZWOCameraError("open() must be called before start_capture()")
        if self._capturing:
            return
        self.camera.start_video_capture()
        self._capturing = True

    def get_frame(self, *, timeout_ms: int = 2000) -> np.ndarray:
        """One ``(H, W)`` frame at the configured bit depth."""
        if self.camera is None or not self._capturing:
            raise ZWOCameraError("start_capture() must be called before get_frame()")
        dtype = np.uint16 if self.bit_depth == 16 else np.uint8
        raw = self.camera.get_video_data(timeout=timeout_ms)
        return np.frombuffer(raw, dtype=dtype).reshape((self.height, self.width))

    def stop_capture(self) -> None:
        if self.camera is not None and self._capturing:
            try:
                self.camera.stop_video_capture()
            except Exception:
                pass
        self._capturing = False

    def auto_adjust_gain(self, frame: np.ndarray, *, step: int = 15, min_gain: int = 0, max_gain: int = 280) -> int:
        """Nudge gain toward mid-brightness, mirroring the source app's preview auto-gain."""
        if self.camera is None:
            return 0
        asi = _load_sdk()
        low, high = _AUTO_GAIN_BAND[self.bit_depth]
        mean = float(np.mean(frame))
        gain = int(self.camera.get_control_value(asi.ASI_GAIN)[0])
        if mean < low and gain < max_gain:
            gain = min(max_gain, gain + step)
        elif mean > high and gain > min_gain:
            gain = max(min_gain, gain - step)
        else:
            return gain
        self.camera.set_control_value(asi.ASI_GAIN, gain)
        return gain

    # -- temperature ----------------------------------------------------------- #

    def poll_temperature(self) -> tuple[float, int] | None:
        """``(sensor_temp_c, cooler_power_percent)``, or None when not open."""
        if self.camera is None:
            return None
        asi = _load_sdk()
        try:
            temp = self.camera.get_control_value(asi.ASI_TEMPERATURE)[0] / 10.0
            power = self.camera.get_control_value(asi.ASI_COOLER_POWER_PERC)[0]
        except Exception:
            return None
        return temp, power

    # -- panel ------------------------------------------------------------- #

    def panel(self, parent: "QWidget | None" = None, on_change=None) -> "QWidget | None":
        from .panel import ZWOCameraPanel

        return ZWOCameraPanel(self, parent=parent, on_change=on_change)
