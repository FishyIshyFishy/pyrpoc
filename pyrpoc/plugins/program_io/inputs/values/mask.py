"""``Mask``: a drawn region, with a label and an id that stays the same through edits; the
rules for the drawn masks, their thumbnail, and their file beside a recording."""

from __future__ import annotations

import base64
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPalette, QPixmap

from pyrpoc.core.program_input import Field, ParameterError
from pyrpoc.utils.file_formats.npz import write_npz


# ``eq=False`` because comparing arrays in a dataclass's ``__eq__`` raises.
@dataclass(frozen=True, eq=False)
class Mask:
    """A drawn region on a ``(height, width)`` grid, lit where ``array`` is non-zero, a
    label saying what it was drawn on, and an ``id`` that stays the same through edits."""

    array: np.ndarray
    label: str
    id: str

    def __post_init__(self) -> None:
        array = np.asarray(self.array, dtype=np.uint8)
        if array.ndim != 2 or 0 in array.shape:
            raise ParameterError(f"a mask must be a 2-D grid, got shape {array.shape}")
        object.__setattr__(self, "array", array)

    @staticmethod
    def new_id() -> str:
        return uuid4().hex[:12]

    def describe(self) -> str:
        """The mask's label and size, in one line."""
        height, width = self.array.shape
        return f"{self.label or 'mask'} ({width} × {height})"

    def on_grid(self, height: int, width: int) -> np.ndarray:
        """The lit pixels as booleans, resized nearest-neighbour to ``(height, width)``."""
        lit = self.array > 0
        if lit.shape == (height, width):
            return lit
        source_h, source_w = lit.shape
        rows = np.minimum((np.arange(height, dtype=np.int64) * source_h) // height, source_h - 1)
        cols = np.minimum((np.arange(width, dtype=np.int64) * source_w) // width, source_w - 1)
        return lit[np.ix_(rows, cols)]

    def to_dict(self) -> dict[str, Any]:
        """The mask as JSON: its id, its label, its shape, and its lit pixels packed one
        bit each and base64-encoded."""
        bits = np.packbits(self.array > 0)
        return {
            "id": self.id,
            "label": self.label,
            "shape": list(self.array.shape),
            "bits": base64.b64encode(bits.tobytes()).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Mask:
        """A mask from ``to_dict``'s JSON, raising ``ParameterError`` if it is not one."""
        if not isinstance(raw, dict):
            raise ParameterError("a mask must be an object with an id, a shape and bits")
        try:
            height, width = (int(size) for size in raw["shape"])
            packed = np.frombuffer(base64.b64decode(raw["bits"]), dtype=np.uint8)
            lit = np.unpackbits(packed, count=height * width).reshape(height, width)
            mask_id = str(raw["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ParameterError(f"a mask's id, shape and bits do not fit together: {exc}") from exc
        return cls(lit, str(raw.get("label", "")), mask_id)


def thumbnail(mask: Mask, size: int) -> QPixmap:
    """A ``size`` square tile: the mask's lit pixels in the accent colour on the input
    background, framed, with the mask's aspect kept."""
    palette = QGuiApplication.palette()
    accent = palette.color(QPalette.ColorRole.Highlight)
    alpha = np.ascontiguousarray(np.where(mask.array > 0, 255, 0).astype(np.uint8))
    height, width = alpha.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = accent.red(), accent.green(), accent.blue()
    rgba[..., 3] = alpha
    shape = QImage(rgba.tobytes(), width, height, 4 * width, QImage.Format.Format_RGBA8888)
    scaled = QPixmap.fromImage(shape.copy()).scaled(
        size - 4, size - 4, Qt.AspectRatioMode.KeepAspectRatio
    )
    tile = QPixmap(size, size)
    tile.fill(palette.color(QPalette.ColorRole.Base))
    painter = QPainter(tile)
    painter.drawPixmap((size - scaled.width()) // 2, (size - scaled.height()) // 2, scaled)
    painter.setPen(QColor(palette.color(QPalette.ColorRole.Mid)))
    painter.drawRect(0, 0, size - 1, size - 1)
    painter.end()
    return tile


@dataclass(frozen=True)
class MasksField(Field):
    """The rules for the drawn masks: a tuple of ``Mask``, in the order they were drawn."""

    def coerce(self, value: Any) -> tuple[Mask, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of masks")
        return tuple(each if isinstance(each, Mask) else Mask.from_dict(each) for each in value)

    def encode(self, value: tuple[Mask, ...]) -> Any:
        return [mask.to_dict() for mask in value]


class MasksNpz:
    """The drawn masks as one ``.npz``: mask ``i``'s pixels as ``mask_<i>``, and every mask's
    ``ids`` and ``labels`` in the same order."""

    suffix = ".npz"

    def write(self, path: Path, value: tuple[Mask, ...]) -> None:
        write_npz(
            path,
            {
                **{f"mask_{index}": mask.array for index, mask in enumerate(value)},
                "ids": np.asarray([mask.id for mask in value], dtype=str),
                "labels": np.asarray([mask.label for mask in value], dtype=str),
            },
        )
