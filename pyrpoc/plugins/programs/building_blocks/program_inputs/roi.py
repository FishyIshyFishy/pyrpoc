"""The region a region scan covers: one drawn mask, or a centred rectangle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PyQt6.QtWidgets import QWidget

from pyrpoc.core.data import Mask
from pyrpoc.core.input_section import InputSection
from pyrpoc.core.library import Library
from pyrpoc.core.program_inputs import (
    Editor,
    Field,
    FieldContext,
    ProgramInputValues,
    declare,
    float_input,
)

from .modulation import CLOSED_SUFFIX, MaskRef, MaskSourceCombo, resolve_mask

# The picker's entry for no mask, which scans the centred rectangle.
CENTER = "Center region"


class RegionMaskCombo(MaskSourceCombo):
    """The open masks, after an entry for the centred rectangle."""

    def __init__(self, library: Library, parent: QWidget):
        super().__init__(lambda: library.matching(Mask), CENTER, parent)
        self.library = library

    def value(self) -> MaskRef | None:
        source_id = self.currentData()
        if not source_id:
            return None
        program_output = self.library.by_id(source_id, Mask)
        label = program_output.label if program_output is not None else self.currentText()
        return MaskRef(source_id=source_id, source_label=label.removesuffix(CLOSED_SUFFIX))

    def set_value(self, mask: MaskRef | None) -> None:
        self.clear()
        self.addItem(CENTER, "")
        if mask is not None:
            self.addItem(mask.describe(), mask.source_id)
            self.setCurrentIndex(1)


@dataclass(frozen=True)
class RegionMaskField(Field):
    """One mask by reference, or None for the centred rectangle."""

    def coerce(self, value: Any) -> MaskRef | None:
        if value is None:
            return None
        return MaskRef.from_value(value)

    def encode(self, value: MaskRef | None) -> Any:
        return None if value is None else value.to_dict()

    def resolve(self, value: MaskRef | None, library: Library) -> MaskRef | None:
        return None if value is None else resolve_mask(value, library)

    def editor(self, parent: Any, context: FieldContext) -> Editor:
        # Narrows for the type checker: every form holding masks offers the library.
        if context.library is None:
            raise ValueError("a mask field needs the open data")
        combo = RegionMaskCombo(context.library, parent)
        return Editor(
            combo,
            get=combo.value,
            set=combo.set_value,
            connect=lambda cb: combo.currentIndexChanged.connect(lambda *_: cb()),
            summary=combo.currentText,
            spec=self,
        )


MASK = declare(
    "roi.mask",
    RegionMaskField(
        "Mask",
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

    mask: MaskRef | None
    center_fraction: float

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> ScanRegion:
        return cls(values[MASK], values[CENTER_FRACTION])
