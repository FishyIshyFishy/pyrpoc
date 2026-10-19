"""Region: the part of the scan grid a region scan covers, either one authored
mask or a centred rectangle. The mask is a library reference, so it has its
own field type and picker, shared with Modulation's table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from PyQt6.QtWidgets import QWidget

from pyrpoc.structs.data_library.data import Mask2D
from pyrpoc.structs.data_library.library import Library
from pyrpoc.structs.plugins.params import (
    Editor,
    Field,
    FieldContext,
    Group,
    block,
    float_field,
    spec_field,
)

from .modulation import CLOSED_SUFFIX, MaskRef, MaskSourceCombo, resolve_mask

# The picker's entry for no mask, which scans the centred rectangle.
CENTER = "Center region"


class RegionMaskCombo(MaskSourceCombo):
    """The open masks, after an entry for the centred rectangle."""

    def __init__(self, library: Library, parent: QWidget):
        super().__init__(lambda: library.matching(Mask2D), CENTER, parent)
        self.library = library

    def value(self) -> MaskRef | None:
        source_id = self.currentData()
        if not source_id:
            return None
        dataset = self.library.by_id(source_id)
        label = dataset.label if dataset is not None else self.currentText()
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


@block
@dataclass
class RoiGroup(Group):
    label: ClassVar[str] = "Region"

    mask: MaskRef | None = spec_field(
        None,
        RegionMaskField(
            "Mask",
            "One connected region drawn in the Mask Editor, or the centred rectangle. "
            "Each row is scanned only across the region, plus the scan's extra steps",
        ),
    )
    center_fraction: float = float_field(
        "Center Fraction",
        0.25,
        minimum=0.01,
        maximum=1.0,
        step=0.05,
        decimals=3,
        tooltip="Width and height of the centred rectangle, as a fraction of the scan, "
        "when no mask is chosen",
    )
