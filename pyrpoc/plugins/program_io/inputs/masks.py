"""``MASKS``, the program input holding every drawn mask, and ``thumbnail``."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QColor,
    QGuiApplication,
    QImage,
    QPainter,
    QPalette,
    QPixmap,
)

from pyrpoc.core.program_input import Field, ParameterError, declare

from .mask import Mask


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


MASKS = declare(
    "masks",
    # Shown by the lines input, which draws each mask beside its line.
    MasksField(
        "Masks", "The masks drawn in a Mask Editor, which is where they are edited", shown=False
    ),
    (),
)
