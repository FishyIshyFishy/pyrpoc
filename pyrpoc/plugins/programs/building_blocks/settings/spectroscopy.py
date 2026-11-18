"""FLIM's histograms, split confocal's windows and hyperspectral's delay sweep, read from their
program inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.core.program_input import InputReader
from pyrpoc.plugins.program_io.inputs import (
    DELAY_START_UM,
    DELAY_STEPS,
    DELAY_STOP_UM,
    FRAME_SETTLE_S,
    HISTOGRAM_BINS,
    HISTOGRAM_BINWIDTH_PS,
    LASER_FREQUENCY_MHZ,
    T0_SAMPLES,
    T1_SAMPLES,
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
    def from_inputs(cls, values: InputReader) -> HistogramSettings:
        return cls(
            values[LASER_FREQUENCY_MHZ],
            values[HISTOGRAM_BINS],
            values[HISTOGRAM_BINWIDTH_PS],
            values[FRAME_SETTLE_S],
        )

    @property
    def laser_period_ps(self) -> int:
        return round(1e6 / self.laser_frequency_mhz)


@dataclass(frozen=True)
class SplitWindows:
    """The samples in each pixel's first window, and those discarded after it."""

    t0_samples: int
    t1_samples: int

    @classmethod
    def from_inputs(cls, values: InputReader) -> SplitWindows:
        return cls(values[T0_SAMPLES], values[T1_SAMPLES])


@dataclass(frozen=True)
class DelaySweep:
    """Where the delay stage starts and stops, and how many images it takes."""

    start_um: float
    stop_um: float
    steps: int

    @classmethod
    def from_inputs(cls, values: InputReader) -> DelaySweep:
        return cls(values[DELAY_START_UM], values[DELAY_STOP_UM], values[DELAY_STEPS])

    def positions_um(self) -> tuple[float, ...]:
        """Each image's stage position, in sweep order; one step is the start alone."""
        return tuple(float(p) for p in np.linspace(self.start_um, self.stop_um, self.steps))
