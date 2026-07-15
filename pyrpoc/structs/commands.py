from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyrpoc.structs.acquired_data import DataKind
from pyrpoc.structs.contexts import MaskContext


@dataclass
class Command:
    """One abstract acquisition step.

    The engine (source/decorator/executor) treats a command opaquely and never
    reads subclass fields; only the matching handler interprets it. Concrete
    variants carry exactly what their handler needs.
    """

    expected_kinds: list[DataKind] = field(default_factory=list)
    frame_index: int = 0


@dataclass
class RasterScanCommand(Command):
    """One galvo raster frame read on the analog inputs (confocal)."""

    device_name: str = "Dev1"
    sample_rate_hz: float = 100_000.0
    fast_axis_ao: int = 0
    slow_axis_ao: int = 1
    x_pixels: int = 512
    y_pixels: int = 512
    extra_left: int = 0
    extra_right: int = 0
    dwell_time_us: float = 2.0
    fast_axis_offset: float = 0.0
    fast_axis_amplitude: float = 1.0
    slow_axis_offset: float = 0.0
    slow_axis_amplitude: float = 1.0
    active_ai_channels: tuple[int, ...] = ()
    channel_labels: list[str] = field(default_factory=list)
    mask_contexts: list[MaskContext] = field(default_factory=list)


@dataclass
class SplitScanCommand(RasterScanCommand):
    """A raster frame time-gated per pixel into t0/t2 halves (split confocal)."""

    t0_samples: int = 1
    t1_samples: int = 0


@dataclass
class PointDwellCommand(Command):
    """Park the galvo at one point and read for a dwell (click-to-acquire)."""

    device_name: str = "Dev1"
    sample_rate_hz: float = 100_000.0
    fast_axis_ao: int = 0
    slow_axis_ao: int = 1
    x_volts: float = 0.0
    y_volts: float = 0.0
    dwell_time_us: float = 2.0
    active_ai_channels: tuple[int, ...] = ()
    channel_labels: list[str] = field(default_factory=list)
    mask_contexts: list[MaskContext] = field(default_factory=list)


@dataclass
class FlimScanCommand(Command):
    """One galvo raster whose photons are histogrammed by the TimeTagger."""

    device_name: str = "Dev1"
    sample_rate_hz: float = 1_000_000.0
    fast_axis_ao: int = 0
    slow_axis_ao: int = 1
    x_pixels: int = 512
    y_pixels: int = 512
    extra_left: int = 0
    extra_right: int = 0
    dwell_time_us: float = 2.0
    fast_axis_offset: float = 0.0
    fast_axis_amplitude: float = 1.0
    slow_axis_offset: float = 0.0
    slow_axis_amplitude: float = 1.0
    frame_trigger_pfi_line: int = 0
    pixel_clock_ctr: int = 0
    pixel_clock_pfi_line: int = 1
    histogram_bins: int = 125
    histogram_binwidth_ps: int = 100
    laser_period_ps: int = 12_500
    # Live hardware handle supplied by the FLIM setup; None when simulated.
    flim: object | None = None
    simulated: bool = False
