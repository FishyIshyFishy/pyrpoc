"""The raster scan: geometry and dwell. Shared by every program that scans the
galvos. ``fast_volts`` and ``slow_volts`` are where a column and a row are
sampled; every waveform and ``voltage_at`` are built on them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from pyrpoc.structs.plugins.params import Group, block, float_field, int_field


@block
@dataclass
class ScanGroup(Group):
    """How the beam is scanned. How many times is ``FrameCountGroup``'s, since
    a mosaic scans once per tile."""

    label: ClassVar[str] = "Scan"

    x_pixels: int = int_field("X Pixels", 512, minimum=8, tooltip="Number of pixels in X")
    y_pixels: int = int_field("Y Pixels", 512, minimum=8, tooltip="Number of pixels in Y")
    extra_left: int = int_field(
        "Extra Steps Left", 300, minimum=0, tooltip="Extra scan steps at the left edge"
    )
    extra_right: int = int_field(
        "Extra Steps Right", 20, minimum=0, tooltip="Extra scan steps at the right edge"
    )
    fast_axis_offset: float = float_field("Fast Axis Offset", 0.0, tooltip="Fast-axis offset")
    fast_axis_amplitude: float = float_field(
        "Fast Axis Amplitude", 1.0, minimum=1e-6, tooltip="Fast-axis amplitude"
    )
    slow_axis_offset: float = float_field("Slow Axis Offset", 0.0, tooltip="Slow-axis offset")
    slow_axis_amplitude: float = float_field(
        "Slow Axis Amplitude", 1.0, minimum=1e-6, tooltip="Slow-axis amplitude"
    )
    dwell_time_us: float = float_field(
        "Dwell Time (us)", 2.0, minimum=0.1, tooltip="Pixel dwell time"
    )

    @property
    def total_x(self) -> int:
        return self.x_pixels + self.extra_left + self.extra_right

    @property
    def kept_columns(self) -> slice:
        """The scanned columns that are displayed, between the overscan."""
        return slice(self.extra_left, self.extra_left + self.x_pixels)

    def fast_volts(self, columns: np.ndarray) -> np.ndarray:
        """Fast-axis volts at displayed ``columns``. Columns left of 0 or past
        ``x_pixels`` are overscan, continuing at the same pitch."""
        fast_step = 2.0 * self.fast_axis_amplitude / self.x_pixels
        return self.fast_axis_offset - self.fast_axis_amplitude + columns * fast_step

    def slow_volts(self, rows: np.ndarray) -> np.ndarray:
        """Slow-axis volts at displayed ``rows``."""
        relative = -1.0 + 2.0 * rows / self.y_pixels
        return self.slow_axis_offset + relative * self.slow_axis_amplitude

    def voltage_at(self, x: int, y: int) -> tuple[float, float]:
        """The (fast, slow) volts at which displayed pixel ``(x, y)`` was sampled.

        Displayed frames are already cropped of overscan, and each pixel is held
        at one voltage for its whole dwell, so there is no half-pixel centre to add.
        """
        return float(self.fast_volts(np.asarray(x))), float(self.slow_volts(np.asarray(y)))
