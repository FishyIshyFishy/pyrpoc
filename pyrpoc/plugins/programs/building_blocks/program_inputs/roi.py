"""The region a region scan covers: one drawn mask, or a centred rectangle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyrpoc.core.input_section import InputSection
from pyrpoc.core.mask import Mask
from pyrpoc.core.program_inputs import (
    Editor,
    Field,
    ProgramInputValues,
    declare,
    float_input,
)

from .mask_cell import MaskCell

# What an empty mask input scans.
CENTER = "Center region"


@dataclass(frozen=True)
class RegionMaskField(Field):
    """One mask, or None for the centred rectangle."""

    accepts = Mask

    def coerce(self, value: Any) -> Mask | None:
        # A region saved before masks were held inline names a library entry,
        # which did not outlive its session: it comes back as the rectangle.
        if value is None or (isinstance(value, dict) and "bits" not in value):
            return None
        return value if isinstance(value, Mask) else Mask.from_dict(value)

    def encode(self, value: Mask | None) -> Any:
        return None if value is None else value.to_dict()

    def editor(self, parent: Any) -> Editor:
        cell = MaskCell(CENTER, parent)
        return Editor(
            cell,
            get=cell.value,
            set=cell.set_value,
            connect=lambda cb: cell.changed.connect(cb),
            summary=cell.summary,
            spec=self,
        )


MASK = declare(
    "roi.mask",
    RegionMaskField(
        "Region mask",
        "One connected region drawn in the Mask Editor, or the centred rectangle. "
        "Each row is scanned only across the region, plus the scan's extra steps",
    ),
    None,
)
CENTER_FRACTION = float_input(
    "roi.center_fraction",
    "Center Fraction",
    0.25,
    minimum=0.01,
    maximum=1.0,
    step=0.05,
    decimals=3,
    tooltip="Width and height of the centred rectangle, as a fraction of the scan, "
    "when no mask is chosen",
)

REGION_SECTION = InputSection("Region", (MASK, CENTER_FRACTION))


@dataclass(frozen=True)
class ScanRegion:
    """The part of the scan grid a region scan covers: a mask, or the centred
    rectangle ``center_fraction`` of the scan wide and high."""

    mask: Mask | None
    center_fraction: float

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> ScanRegion:
        return cls(values[MASK], values[CENTER_FRACTION])
