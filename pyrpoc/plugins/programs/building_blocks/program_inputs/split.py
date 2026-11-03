"""How split confocal divides each pixel's samples into windows."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.input_section import InputSection
from pyrpoc.core.program_inputs import ProgramInputValues, int_input

T0_SAMPLES = int_input(
    "split.t0_samples",
    "t0 Samples",
    1,
    minimum=1,
    tooltip="Number of samples in the first subpixel window",
)
T1_SAMPLES = int_input(
    "split.t1_samples",
    "t1 Samples",
    0,
    minimum=0,
    tooltip="Number of samples to discard between t0 and t2",
)

SPLIT_SECTION = InputSection("Split", (T0_SAMPLES, T1_SAMPLES))


@dataclass(frozen=True)
class SplitWindows:
    """The samples in each pixel's first window, and those discarded after it."""

    t0_samples: int
    t1_samples: int

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> SplitWindows:
        return cls(values[T0_SAMPLES], values[T1_SAMPLES])
