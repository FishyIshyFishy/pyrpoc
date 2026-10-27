"""Treatment: the mask the treatment laser followed, by reference to its
library entry. It must be set, so it has its own field type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pyrpoc.structs.data_library.data import Mask2D
from pyrpoc.structs.data_library.library import Library
from pyrpoc.structs.plugins.params import (
    Editor,
    Field,
    FieldContext,
    Group,
    ParameterError,
    block,
    spec_field,
)

from .library_ref import RefCombo
from .modulation import MASK_TOOLTIP, MaskRef, resolve_mask

NO_MASK = "Choose a mask"


@dataclass(frozen=True)
class TreatedMaskField(Field):
    """One mask, which must be set: without it nothing counts as treated."""

    def coerce(self, value: Any) -> MaskRef | None:
        if value is None:
            return None
        return MaskRef.from_value(value)

    def encode(self, value: MaskRef | None) -> Any:
        return None if value is None else value.to_dict()

    def resolve(self, value: MaskRef | None, library: Library) -> MaskRef:
        if value is None:
            raise ParameterError(f"{self.label}: choose the mask the treatment laser followed")
        return resolve_mask(value, library)

    def editor(self, parent: Any, context: FieldContext) -> Editor:
        # Narrows for the type checker: every program form offers the library.
        if context.library is None:
            raise ValueError("a mask field needs the open data")
        library = context.library
        combo = RefCombo(
            library, lambda: library.matching(Mask2D), MaskRef, NO_MASK, MASK_TOOLTIP, parent
        )
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
class TreatmentGroup(Group):
    label: ClassVar[str] = "Treatment"

    mask: MaskRef | None = spec_field(
        None,
        TreatedMaskField(
            "Treated Mask",
            "The mask the treatment laser followed. Fluorescent pixels inside it are "
            "treated; those outside are the control",
        ),
    )
