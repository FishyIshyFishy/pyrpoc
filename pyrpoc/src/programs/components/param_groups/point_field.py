"""Points: where the galvos park, and which pixel said so."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyrpoc.src.structs.data import Dataset
from pyrpoc.src.structs.params import (
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

    ``fast_v``/``slow_v`` are what the hardware does, so a run reproduces from
    its saved parameters alone with no reference to the image it came from.
    ``source_id`` and the pixel are provenance on top of that: they say this
    spectrum is pixel (37, 204) of a particular run, which is worth the fields.
    A point typed by hand has no source, which is what the defaults mean.

    ``source_label`` is the human half of that identity, and it is stored rather
    than resolved for the same reason ``Provenance.name`` is: a dataset id means
    nothing to anyone reading the metadata six months later, and the dataset it
    named may not be open any more.

    Voltages rather than pixels are the parameter because a pixel is only
    meaningful against a scan geometry, and that geometry belongs to the image
    -- not to the program being configured.
    """

    fast_v: float = 0.0
    slow_v: float = 0.0
    source_id: str = ""
    source_label: str = ""
    pixel_x: int = -1
    pixel_y: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(self, "fast_v", float(self.fast_v))
        object.__setattr__(self, "slow_v", float(self.slow_v))
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "source_label", str(self.source_label))
        object.__setattr__(self, "pixel_x", int(self.pixel_x))
        object.__setattr__(self, "pixel_y", int(self.pixel_y))

    @classmethod
    def from_pixel(cls, dataset: Dataset, x: int, y: int) -> Point:
        """The position pixel ``(x, y)`` of ``dataset`` was measured at.

        The geometry comes from the dataset's provenance rather than the live
        ``ScanGroup``. Blocks are shared and mutable: change the amplitude after
        taking an image and the live block no longer describes the picture being
        clicked, so the volts would point somewhere it never looked.
        """
        raw = dataset.provenance.parameters.get(block_name(ScanGroup))
        if not isinstance(raw, dict):
            raise ParameterError(
                f"{dataset.label} was not acquired with a scan geometry, "
                "so a pixel does not name a position"
            )
        try:
            scan = decode_block(ScanGroup, raw)
            fast_v, slow_v = scan.voltage_at(x, y)
        except Exception as exc:  # noqa: BLE001 - a bad record, reported as one
            raise ParameterError(f"could not place that pixel: {exc}") from exc
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
    def from_dict(cls, raw: Any) -> Point:
        if isinstance(raw, Point):
            return raw
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ParameterError("a point must be an object with fast_v/slow_v")
        return cls(
            fast_v=float(raw.get("fast_v", 0.0)),
            slow_v=float(raw.get("slow_v", 0.0)),
            source_id=str(raw.get("source_id", "")),
            source_label=str(raw.get("source_label", "")),
            pixel_x=int(raw.get("pixel_x", -1)),
            pixel_y=int(raw.get("pixel_y", -1)),
        )


@dataclass(frozen=True)
class PointField(Field):
    """A galvo position: volts, plus the pixel they were picked from, if any.

    ``coerce`` has to be idempotent on a live ``Point``: the form coerces what
    its widget hands back, and ``Executor.start`` re-coerces every held value
    through ``BlockStore.validate`` before a run begins. ``decode`` is inherited
    -- ``Field.decode`` is ``coerce``, and ``from_dict`` takes both forms.
    """

    def coerce(self, value: Any) -> Point:
        return Point.from_dict(value)

    def encode(self, value: Any) -> Any:
        return Point.from_dict(value).to_dict()

    def editor(self, parent: Any, context: FieldContext) -> Editor:
        from ..editors import point_editor

        return point_editor(parent, context)


def point_field(label="Target", *, tooltip=""):
    return spec_field(None, PointField(label, tooltip), factory=Point)
