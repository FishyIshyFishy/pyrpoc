"""The simulated slide: how large a pixel is on it, and how noisy the detector is."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.utils.value_rules import float_input

from .synthetic_detector import NOISE_LEVEL

UM_PER_PIXEL = float_input(
    "um_per_pixel",
    "Pixel Size (um)",
    0.5,
    minimum=0.001,
    step=0.1,
    decimals=4,
    tooltip="Stage distance one pixel spans; with the spacing, sets the tile overlap",
)


@dataclass(frozen=True)
class SyntheticSlide:
    """How large a pixel is on the simulated slide, standing in for the
    objective, and how noisy the detector is."""

    um_per_pixel: float
    noise_level: float

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> SyntheticSlide:
        return cls(values[UM_PER_PIXEL], values[NOISE_LEVEL])
