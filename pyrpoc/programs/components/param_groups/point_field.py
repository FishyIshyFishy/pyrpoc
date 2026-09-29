"""Points: where the galvos park, and which pixel said so."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyrpoc.structs.data import Dataset
from pyrpoc.structs.params import (
    Editor,
    Field,
    FieldContext,
    ParameterError,
    block_name,
    decode_block,
    spec_field,
)

from .scan import ScanGroup


@dataclass(frozen=True)
class Point:
    """One parked galvo position.

    Volts are the parameter, so a run reproduces from its saved parameters
    alone: a pixel only means something against the scan geometry of the image
    it was picked from. The source and pixel are provenance on top; a point
    typed by hand has none, which is what the defaults mean.
    """

    fast_v: float = 0.0
    slow_v: float = 0.0
    source_id: str = ""
    source_label: str = ""
    pixel_x: int = -1
    pixel_y: int = -1

    @classmethod
    def from_pixel(cls, dataset: Dataset, x: int, y: int) -> Point:
        """The position pixel ``(x, y)`` of ``dataset`` was measured at. The
        geometry comes from the dataset's provenance, not the live block, which
        may have been edited since the image was taken."""
        raw = dataset.provenance.parameters.get(block_name(ScanGroup))
        if raw is None:
            raise ParameterError(
                f"{dataset.label} was not acquired with a scan geometry, "
                "so a pixel does not name a position"
            )
        fast_v, slow_v = decode_block(ScanGroup, raw).voltage_at(x, y)
        return cls(fast_v, slow_v, dataset.id, dataset.label, x, y)

    @property
    def picked(self) -> bool:
        """Whether this point was picked from data rather than typed."""
        return bool(self.source_id) and self.pixel_x >= 0 and self.pixel_y >= 0

    def describe(self) -> str:
        """Where this point came from, for the label under the spin boxes."""
        if not self.picked:
            return "typed"
        return f"{self.source_label or self.source_id} px ({self.pixel_x}, {self.pixel_y})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fast_v": self.fast_v,
            "slow_v": self.slow_v,
            "source_id": self.source_id,
            "source_label": self.source_label,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
        }

    @classmethod
    def from_value(cls, raw: Any) -> Point:
        """A point from the form (already a ``Point``) or from JSON (a dict)."""
        if isinstance(raw, Point):
            return raw
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ParameterError("a point must be an object with fast_v/slow_v")
        try:
            return cls(
                fast_v=float(raw.get("fast_v", 0.0)),
                slow_v=float(raw.get("slow_v", 0.0)),
                source_id=str(raw.get("source_id", "")),
                source_label=str(raw.get("source_label", "")),
                pixel_x=int(raw.get("pixel_x", -1)),
                pixel_y=int(raw.get("pixel_y", -1)),
            )
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"a point's volts and pixel must be numbers: {exc}") from exc


@dataclass(frozen=True)
class PointField(Field):
    """A galvo position: volts, plus the pixel they were picked from, if any.
    ``coerce`` takes a live ``Point`` as well as JSON, since the executor
    re-coerces every held value before a run."""

    def coerce(self, value: Any) -> Point:
        return Point.from_value(value)

    def encode(self, value: Point) -> Any:
        return value.to_dict()

    def editor(self, parent: Any, context: FieldContext) -> Editor:
        from ..editors import point_editor

        del context
        return point_editor(self, parent)


def point_field(label="Target", *, tooltip=""):
    return spec_field(None, PointField(label, tooltip), factory=Point)
