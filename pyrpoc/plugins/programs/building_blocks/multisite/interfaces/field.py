"""One field of view on the dish as a scan: its volts from the calibration, and its frames
turned to lie as the dish does."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.plugins.program_io.inputs import ScanGeometry


@dataclass(frozen=True)
class FieldOfView:
    """A square field of view: how wide on the dish, at how many pixels a side, the
    calibration's signed µm per volt, the dwell, and the rig's overscan."""

    field_um: float
    pixels: int
    um_per_volt: tuple[float, float]
    dwell_us: float
    extra_left: int
    extra_right: int

    @property
    def um_per_px(self) -> float:
        return self.field_um / self.pixels

    def around(self, pixels: int) -> FieldOfView:
        """A square field ``pixels`` a side at the same pixel size, centred where this one is."""
        return FieldOfView(
            pixels * self.um_per_px,
            pixels,
            self.um_per_volt,
            self.dwell_us,
            self.extra_left,
            self.extra_right,
        )

    def scan(self) -> ScanGeometry:
        """The field centred on the stage's position."""
        fast_amplitude = self.field_um / 2.0 / abs(self.um_per_volt[0])
        slow_amplitude = self.field_um / 2.0 / abs(self.um_per_volt[1])
        # Pixel i is sampled (i - pixels / 2) pixels from the stage. Reversed by ``on_dish``, an
        # axis running against the stage would put the stage at pixels / 2 - 1, so its scan is
        # moved one pixel to keep the stage at pixels / 2 either way.
        fast_offset = 2.0 * fast_amplitude / self.pixels if self.um_per_volt[0] < 0 else 0.0
        slow_offset = 2.0 * slow_amplitude / self.pixels if self.um_per_volt[1] < 0 else 0.0
        return ScanGeometry(
            x_pixels=self.pixels,
            y_pixels=self.pixels,
            extra_left=self.extra_left,
            extra_right=self.extra_right,
            fast_axis_offset=fast_offset,
            fast_axis_amplitude=fast_amplitude,
            slow_axis_offset=slow_offset,
            slow_axis_amplitude=slow_amplitude,
            dwell_time_us=self.dwell_us,
        )

    def on_dish(self, frame: np.ndarray) -> np.ndarray:
        """A ``(C, H, W)`` frame of this field turned so x and y grow as they do on the
        dish."""
        # A negative scale means the scan runs against the stage along that axis.
        if self.um_per_volt[0] < 0:
            frame = frame[:, :, ::-1]
        if self.um_per_volt[1] < 0:
            frame = frame[:, ::-1, :]
        return np.ascontiguousarray(frame)

    def corner_um(self, x_um: float, y_um: float) -> tuple[float, float]:
        """The dish position of the top-left pixel when the stage is at (``x_um``, ``y_um``)."""
        return x_um - self.field_um / 2.0, y_um - self.field_um / 2.0
