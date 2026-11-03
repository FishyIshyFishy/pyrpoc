"""The simulated slide: how large a pixel is on it, and how noisy the detector is."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.input_section import InputSection
from pyrpoc.core.program_inputs import ProgramInputValues, float_input

UM_PER_PIXEL = float_input(
    "specimen.um_per_pixel",
    "Pixel Size (um)",
    0.5,
    minimum=0.001,
    step=0.1,
    decimals=4,
    tooltip="Stage distance one pixel spans; with the spacing, sets the tile overlap",
)
NOISE_LEVEL = float_input(
    "specimen.noise_level",
    "Noise Level",
    0.03,
    minimum=0.0,
    step=0.01,
    tooltip="Gaussian noise added per pixel",
)

SPECIMEN_SECTION = InputSection("Specimen", (UM_PER_PIXEL, NOISE_LEVEL))


@dataclass(frozen=True)
class SyntheticSlide:
    """How large a pixel is on the simulated slide, standing in for the
    objective, and how noisy the detector is."""

    um_per_pixel: float
    noise_level: float

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> SyntheticSlide:
        return cls(values[UM_PER_PIXEL], values[NOISE_LEVEL])
