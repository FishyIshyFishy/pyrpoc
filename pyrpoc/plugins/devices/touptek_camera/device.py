"""A ToupTek USB camera, such as the brightfield U3CMOS08500KPA.

The session stays open between runs; the app opens it at launch and closes it
on exit. Only one process can hold the camera, so ToupView cannot use it
while pyrpoc is running. Exposure and gain belong to the program, not here:
this block only says which camera to open.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyrpoc.structs.plugins import params as P
from pyrpoc.structs.plugins.devices import Device, device_registry

from .sdk import ToupcamError, ToupcamSession, row_pitch, unpack_rgb24

if TYPE_CHECKING:  # pragma: no cover
    from PyQt6.QtWidgets import QWidget

# How often a grab checks whether a frame has arrived.
POLL_S = 0.01
# Time allowed past the exposure for a frame to cross USB.
FRAME_MARGIN_S = 2.0


@dataclass
class ToupcamConfig(P.Group):
    camera: str = P.text_field(
        "Camera",
        "",
        tooltip="Name (e.g. U3CMOS08500KPA) or id of the camera to open; blank opens the first",
    )


@device_registry.register("touptek_camera")
class ToupcamCamera(Device):
    display_name = "ToupTek Camera"
    owns_connection = True
    holds_session = True
    config_cls = ToupcamConfig

    config: ToupcamConfig

    def __init__(self, instance_id: str | None = None, user_label: str | None = None):
        super().__init__(instance_id=instance_id, user_label=user_label)
        self.session: ToupcamSession | None = None
        self.exposure_s = 0.0

    def summary(self) -> str:
        if self.session is None:
            return "not connected"
        return f"{self.session.camera.name} - connected"

    @property
    def session_open(self) -> bool:
        return self.session is not None

    def open_session(self) -> None:
        if self.session is not None:
            raise ToupcamError(f"{self.name} is already connected")
        self.session = ToupcamSession.connect(self.config.camera)

    def close_session(self) -> None:
        # Dropped first, so a camera that faults on close still leaves the
        # device closed rather than half-open.
        session = self.require_session()
        self.session = None
        session.close()

    def check_reachable(self) -> bool:
        self.require_session().frame_size()
        return True

    def require_session(self) -> ToupcamSession:
        if self.session is None:
            raise ToupcamError(f"{self.name} is not connected; connect it in the Devices panel")
        return self.session

    def start(self, *, exposure_us: int, gain_pct: int, binned: bool) -> None:
        """Apply the settings and start streaming. Values outside what the
        camera accepts are refused with its range rather than clamped."""
        session = self.require_session()
        low, high = session.exposure_range_us()
        if not low <= exposure_us <= high:
            raise ToupcamError(f"exposure must be {low}-{high} µs, got {exposure_us}")
        low, high = session.gain_range_pct()
        if not low <= gain_pct <= high:
            raise ToupcamError(f"gain must be {low}-{high} %, got {gain_pct}")
        # Index 1 is the camera's 2x2-binned size; index 0 is full frame.
        session.start(1 if binned else 0, exposure_us, gain_pct)
        self.exposure_s = exposure_us / 1e6

    def grab(self, sleep: Callable[[float], None]) -> np.ndarray:
        """The next frame as ``(H, W, 3)`` uint8, in the SDK's channel order.
        A program passes ``ctx.sleep`` so a stopped run interrupts the wait;
        its ``finally`` should then call ``stop()``."""
        session = self.require_session()
        width, height = session.frame_size()
        buffer = np.empty(row_pitch(width) * height, dtype=np.uint8)
        deadline = time.monotonic() + self.exposure_s + FRAME_MARGIN_S
        while True:
            if session.frame_ready.is_set():
                session.frame_ready.clear()
                if session.fault:
                    raise ToupcamError(session.fault)
                size = session.pull(buffer)
                if size is not None:
                    return unpack_rgb24(buffer, *size)
            if time.monotonic() > deadline:
                raise ToupcamError(f"no frame from {self.name} within the exposure time")
            sleep(POLL_S)

    def stop(self) -> None:
        if self.session is not None:
            self.session.stop()

    def panel(self, parent: QWidget, on_change: Callable[[], None]) -> QWidget:
        from .panel import ToupcamPanel

        return ToupcamPanel(self, parent, on_change)
