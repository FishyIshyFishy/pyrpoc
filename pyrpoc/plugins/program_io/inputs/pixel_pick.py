"""``PixelPick``: the last click on one pixel of an image, and ``PICK``, the program input
holding it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyrpoc.core.program_input import Field, ParameterError, declare
from pyrpoc.plugins.program_io.outputs.image import ImageOutput


@dataclass(frozen=True)
class PixelPick:
    """The pixel clicked, ``x`` along the fast axis and ``y`` along the slow, and the image
    it was clicked on: its id, its label and the inputs it was acquired with."""

    source_id: str
    source_label: str
    parameters: dict[str, Any]
    x: int
    y: int

    @classmethod
    def on(cls, output: ImageOutput, x: int, y: int) -> PixelPick:
        """A pick of pixel ``(x, y)`` of ``output``."""
        return cls(output.id, output.label, dict(output.provenance.parameters), x, y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_label": self.source_label,
            "parameters": self.parameters,
            "x": self.x,
            "y": self.y,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> PixelPick:
        """A pick from ``to_dict``'s JSON, raising ``ParameterError`` if it is not one."""
        if not isinstance(raw, dict) or not isinstance(raw.get("parameters"), dict):
            raise ParameterError("a pick must be an object with a source, parameters and a pixel")
        try:
            return cls(
                str(raw["source_id"]),
                str(raw["source_label"]),
                raw["parameters"],
                int(raw["x"]),
                int(raw["y"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ParameterError(f"a pick's source and pixel do not fit together: {exc}") from exc


@dataclass(frozen=True)
class PickField(Field):
    """The rules for a pick: a ``PixelPick``, or None before any pixel is clicked."""

    def coerce(self, value: Any) -> PixelPick | None:
        if value is None or isinstance(value, PixelPick):
            return value
        return PixelPick.from_dict(value)

    def encode(self, value: PixelPick | None) -> Any:
        return None if value is None else value.to_dict()


# No program shows it: image panels write it, and a runner turns it into what its program reads.
PICK = declare("pick", PickField("Pick", "The last pixel clicked on an image"), None)
