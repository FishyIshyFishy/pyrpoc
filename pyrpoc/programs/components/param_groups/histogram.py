"""FLIM's decay histogram."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.params import Group, float_field, int_field
from pyrpoc.structs.registries import block


@block
@dataclass
class HistogramGroup(Group):
    label: ClassVar[str] = "Histogram"

    laser_frequency_mhz: float = float_field(
        "Laser Frequency MHz", 80.0, minimum=0.001, tooltip="Laser repetition rate in MHz"
    )
    histogram_bins: int = int_field(
        "Histogram Bins", 125, minimum=2, tooltip="Number of decay-histogram bins per pixel"
    )
    histogram_binwidth_ps: int = int_field(
        "Histogram Bin Width (ps)",
        100,
        minimum=1,
        tooltip="Bin width in ps (bins x width should span one laser period)",
    )
    frame_settle_s: float = float_field(
        "Frame Settle (s)",
        5e-3,
        minimum=0.0,
        step=1e-3,
        tooltip="Wait after the scan so the last photons reach the measurement",
    )

    @property
    def laser_period_ps(self) -> int:
        return int(round(1e6 / self.laser_frequency_mhz))
