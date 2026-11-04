"""FLIM's decay histograms: the laser, the bins, and the settle after a frame."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.utils.value_rules import float_input, int_input

LASER_FREQUENCY_MHZ = float_input(
    "laser_frequency_mhz",
    "Laser Frequency MHz",
    80.0,
    minimum=0.001,
    tooltip="Laser repetition rate in MHz",
)
HISTOGRAM_BINS = int_input(
    "histogram_bins",
    "Histogram Bins",
    125,
    minimum=2,
    tooltip="Number of decay-histogram bins per pixel",
)
HISTOGRAM_BINWIDTH_PS = int_input(
    "histogram_binwidth_ps",
    "Histogram Bin Width (ps)",
    100,
    minimum=1,
    tooltip="Bin width in ps (bins x width should span one laser period)",
)
FRAME_SETTLE_S = float_input(
    "frame_settle_s",
    "Frame Settle (s)",
    5e-3,
    minimum=0.0,
    step=1e-3,
    tooltip="Wait after the scan so the last photons reach the measurement",
)


@dataclass(frozen=True)
class HistogramSettings:
    """The laser's repetition rate, the decay histogram's bins, and the settle
    time after each frame."""

    laser_frequency_mhz: float
    histogram_bins: int
    histogram_binwidth_ps: int
    frame_settle_s: float

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> HistogramSettings:
        return cls(
            values[LASER_FREQUENCY_MHZ],
            values[HISTOGRAM_BINS],
            values[HISTOGRAM_BINWIDTH_PS],
            values[FRAME_SETTLE_S],
        )

    @property
    def laser_period_ps(self) -> int:
        return round(1e6 / self.laser_frequency_mhz)
