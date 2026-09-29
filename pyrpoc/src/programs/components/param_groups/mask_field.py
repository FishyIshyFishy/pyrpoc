"""Masks: an authored region plus the digital line it drives, bound by reference."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from typing import Any

import numpy as np

from pyrpoc.src.structs.params import (
    Editor,
    Field,
    FieldContext,
    ParameterError,
    spec_field,
)


@dataclass(frozen=True)
class Mask:
    """One authored mask wired to one digital output line -- by reference.

    The parameter is which library entry, and which port and line it drives.
    ``source_label`` is stored beside the id because a dataset id means nothing
    to anyone reading the metadata six months later, and the entry it named may
    not be open any more.

    ``array`` is empty in the stored value and filled in only when a run starts:
    ``MasksField.resolve`` reads the pixels out of the library into a copy the
    program is handed. A program has no library -- ``RunContext`` hands out
    this run's own parameters and datasets and nothing else -- so the pixels
    have to arrive with the parameters, but the shared block, the session file
    and the run metadata keep only the reference. That is what lets a binding
    survive a relaunch, and what makes a closed entry stop the run instead of
    running silently without it.

    ``array`` is ``compare=False`` because this is a frozen dataclass: the
    generated ``__eq__`` would compare two arrays elementwise and then call
    ``bool()`` on the result, which raises.
    """

    source_id: str = ""
    source_label: str = ""
    array: np.ndarray | None = dc_field(default=None, compare=False)
    port: int = 0
    line: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "source_label", str(self.source_label))
        object.__setattr__(self, "port", int(self.port))
        object.__setattr__(self, "line", int(self.line))
        if self.array is not None:
            array = np.asarray(self.array)
            if array.ndim != 2:
                raise ParameterError(f"a mask must be 2D, got shape={array.shape}")
            object.__setattr__(self, "array", array)

    @property
    def resolved(self) -> bool:
        """Whether this mask carries its pixels, i.e. it is a run's copy."""
        return self.array is not None

    def describe(self) -> str:
        """Which library entry this is, for a row that has lost it."""
        return self.source_label or self.source_id or "no mask"

    def channel(self, device_name: str) -> str:
        """The NI-DAQ channel string this mask drives."""
        return f"{device_name}/port{self.port}/line{self.line}"

    def to_dict(self) -> dict[str, Any]:
        """Provenance only. The array is data, and this is a parameter."""
        return {
            "source_id": self.source_id,
            "source_label": self.source_label,
            "port": self.port,
            "line": self.line,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Mask:
        if isinstance(raw, Mask):
            return raw
        if not isinstance(raw, dict):
            raise ParameterError("a mask must be an object with source_id/port/line")
        return cls(
            source_id=str(raw.get("source_id", "")),
            source_label=str(raw.get("source_label", "")),
            port=int(raw.get("port", 0)),
            line=int(raw.get("line", 0)),
        )


@dataclass(frozen=True)
class MasksField(Field):
    """The Modulation table: library entry, port, line — one row per mask."""

    def coerce(self, value: Any) -> tuple[Mask, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of masks")
        return tuple(Mask.from_dict(row) for row in value)

    def encode(self, value: Any) -> Any:
        return [mask.to_dict() for mask in (value or ())]

    def resolve(self, value: Any, library: Any) -> tuple[Mask, ...]:
        """Each binding with its pixels, read from the open data.

        A binding whose entry is not open refuses the run rather than being
        skipped: acquiring without a mask the user bound is worse than not
        acquiring.
        """
        out: list[Mask] = []
        for mask in self.coerce(value):
            dataset = library.by_id(mask.source_id) if library is not None else None
            array = dataset.latest() if dataset is not None else None
            if array is None:
                raise ParameterError(f"mask '{mask.describe()}' is not open")
            out.append(replace(mask, array=array))
        return tuple(out)

    def editor(self, parent: Any, context: FieldContext) -> Editor:
        from ..editors import mask_editor

        return mask_editor(parent, context)


def masks_field(label="Masks", *, tooltip=""):
    return spec_field(None, MasksField(label, tooltip), factory=tuple)
