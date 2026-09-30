"""Masks: an authored region plus the digital line it drives, bound by reference."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from typing import Any

import numpy as np

from pyrpoc.structs.library import Library
from pyrpoc.structs.params import Editor, Field, FieldContext, ParameterError, spec_field


@dataclass(frozen=True)
class Mask:
    """One authored mask wired to one digital output line, by reference.

    The stored value names a library entry; ``source_label`` sits beside the id
    because an id means nothing in metadata read months later. ``array`` is
    filled only in a run's copy, by ``MasksField.resolve``: a program has no
    library, so the pixels arrive with its parameters while the shared block
    and workspace keep only the reference. ``compare=False`` because comparing
    arrays in a frozen dataclass's ``__eq__`` raises.
    """

    source_id: str = ""
    source_label: str = ""
    array: np.ndarray | None = dc_field(default=None, compare=False)
    port: int = 0
    line: int = 0

    def describe(self) -> str:
        """Which library entry this is, for a row that has lost it."""
        return self.source_label or self.source_id or "no mask"

    def channel(self, device_name: str) -> str:
        """The NI-DAQ channel string this mask drives."""
        return f"{device_name}/port{self.port}/line{self.line}"

    def to_dict(self) -> dict[str, Any]:
        """Provenance only: the array is data, and this is a parameter."""
        return {
            "source_id": self.source_id,
            "source_label": self.source_label,
            "port": self.port,
            "line": self.line,
        }

    @classmethod
    def from_value(cls, raw: Any) -> Mask:
        """A mask from the form (already a ``Mask``) or from JSON (a dict)."""
        if isinstance(raw, Mask):
            return raw
        if not isinstance(raw, dict):
            raise ParameterError("a mask must be an object with source_id/port/line")
        try:
            return cls(
                source_id=str(raw.get("source_id", "")),
                source_label=str(raw.get("source_label", "")),
                port=int(raw.get("port", 0)),
                line=int(raw.get("line", 0)),
            )
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"a mask's port and line must be integers: {exc}") from exc


@dataclass(frozen=True)
class MasksField(Field):
    """The Modulation table: library entry, port, line, one row per mask."""

    def coerce(self, value: Any) -> tuple[Mask, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of masks")
        return tuple(Mask.from_value(row) for row in value)

    def encode(self, value: tuple[Mask, ...]) -> Any:
        return [mask.to_dict() for mask in value]

    def resolve(self, value: tuple[Mask, ...], library: Library) -> tuple[Mask, ...]:
        """Each binding with its pixels. One whose entry is not open refuses the
        run: acquiring without a mask the user bound is worse than not acquiring."""
        out: list[Mask] = []
        for mask in value:
            dataset = library.by_id(mask.source_id)
            array = dataset.latest() if dataset is not None else None
            if array is None:
                raise ParameterError(f"mask '{mask.describe()}' is not open")
            out.append(replace(mask, array=array))
        return tuple(out)

    def editor(self, parent: Any, context: FieldContext) -> Editor:
        from ..editors import mask_editor

        return mask_editor(self, parent, context)


def masks_field(label="Masks", *, tooltip=""):
    return spec_field(None, MasksField(label, tooltip), factory=tuple)
