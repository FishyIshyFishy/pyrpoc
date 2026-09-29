"""The raster scan: geometry, dwell and frame count. Shared across modalities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.src.structs.params import Group, float_field, int_field
from pyrpoc.src.structs.registries import block


@block
@dataclass
class ScanGroup(Group):
    """How the beam is scanned, and how many times. The frame count is here
    because it decides how the imaging is done, like the geometry."""

    label: ClassVar[str] = "Scan"

    num_frames: int = int_field("Frames", 1, minimum=1, tooltip="Number of frames to capture")
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

    def voltage_at(self, x: int, y: int) -> tuple[float, float]:
        """The (fast, slow) volts at which displayed pixel ``(x, y)`` was sampled.

        The inverse of the raster waveform's axes. Displayed frames are already
        cropped of overscan, and each pixel is held at one voltage for its whole
        dwell, so there is no half-pixel centre to add.
        """
        fast_step = 2.0 * self.fast_axis_amplitude / self.x_pixels
        fast_v = self.fast_axis_offset - self.fast_axis_amplitude + x * fast_step
        slow_v = self.slow_axis_offset + (-1.0 + 2.0 * y / self.y_pixels) * self.slow_axis_amplitude
        return fast_v, slow_v
