"""The region a region scan covers: the drawn masks together, or a centred rectangle when
none are drawn."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.input_section import InputSection
from pyrpoc.core.mask import MASKS, Mask
from pyrpoc.core.program_inputs import ProgramInputValues, float_input

CENTER_FRACTION = float_input(
    "roi.center_fraction",
    "Center Fraction",
    0.25,
    minimum=0.01,
    maximum=1.0,
    step=0.05,
    decimals=3,
    tooltip="Width and height of the centred rectangle, as a fraction of the scan, "
    "when no mask is drawn",
)

REGION_SECTION = InputSection("Region", (MASKS, CENTER_FRACTION))


@dataclass(frozen=True)
class ScanRegion:
    """The part of the scan grid a region scan covers: every drawn mask together, or with
    none, the centred rectangle ``center_fraction`` of the scan wide and high."""

    masks: tuple[Mask, ...]
    center_fraction: float

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> ScanRegion:
        return cls(values[MASKS], values[CENTER_FRACTION])
