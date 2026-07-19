"""Shared scan geometry for galvo-raster instruments (real and simulated)."""

from __future__ import annotations

from attrs import define, field


@define
class RasterScan:
    """Parameters for one galvo raster scan, shared by NIDAQ and SimulatedDAQ."""

    x_pixels: int
    y_pixels: int
    dwell_time_us: float
    sample_rate_hz: float
    ai_channels: list[int]
    fast_axis_ao: int = 0
    slow_axis_ao: int = 1
    extra_left: int = 0
    extra_right: int = 0
    fast_axis_offset: float = 0.0
    fast_axis_amplitude: float = 1.0
    slow_axis_offset: float = 0.0
    slow_axis_amplitude: float = 1.0

    @property
    def pixel_samples(self) -> int:
        return max(1, int(self.dwell_time_us * 1e-6 * self.sample_rate_hz))

    @property
    def total_x(self) -> int:
        return self.x_pixels + self.extra_left + self.extra_right
