"""``MASKS``, the program input holding every drawn mask, and its read-only tiles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QColor,
    QFontMetrics,
    QGuiApplication,
    QImage,
    QPainter,
    QPalette,
    QPixmap,
    QResizeEvent,
)
from PyQt6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from pyrpoc.core.program_input import Editor, Field, FieldValues, ParameterError, declare

from .mask import Mask

THUMBNAIL_PX = 48
# A tile's width: its thumbnail and room for the name under it.
TILE_PX = 76


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


class MaskList(QWidget):
    """Shows the masks it is given as tiles, as many to a row as fit; they are drawn and
    edited in a Mask Editor."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.masks: tuple[Mask, ...] = ()
        self.tiles: list[QWidget] = []
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 2, 0, 2)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(6)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.empty = QLabel("None drawn. Draw one in a Mask Editor.", self)
        self.empty.setEnabled(False)
        self.set_value(())

    def value(self) -> tuple[Mask, ...]:
        return self.masks

    def set_value(self, masks: tuple[Mask, ...]) -> None:
        self.masks = masks
        for tile in self.tiles:
            tile.hide()
            tile.deleteLater()
        self.tiles = [self.build_tile(mask) for mask in masks]
        self.empty.setVisible(not masks)
        self.reflow()

    def build_tile(self, mask: Mask) -> QWidget:
        tile = QWidget(self)
        tile.setFixedWidth(TILE_PX)
        tile.setToolTip(mask.describe())
        layout = QVBoxLayout(tile)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        picture = QLabel(tile)
        picture.setPixmap(thumbnail(mask, THUMBNAIL_PX))
        layout.addWidget(picture, 0, Qt.AlignmentFlag.AlignHCenter)
        name = QLabel(tile)
        metrics = QFontMetrics(name.font())
        name.setText(metrics.elidedText(mask.label or "mask", Qt.TextElideMode.ElideRight, TILE_PX))
        layout.addWidget(name, 0, Qt.AlignmentFlag.AlignHCenter)
        return tile

    def reflow(self) -> None:
        """Lay the tiles out in as many columns as the width holds."""
        while self.grid.count():
            self.grid.takeAt(0)
        if not self.tiles:
            self.grid.addWidget(self.empty, 0, 0)
            return
        spacing = self.grid.horizontalSpacing()
        columns = max(1, (self.width() + spacing) // (TILE_PX + spacing))
        for index, tile in enumerate(self.tiles):
            self.grid.addWidget(tile, index // columns, index % columns)

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self.reflow()

    def summary(self) -> str:
        count = len(self.masks)
        return "none" if not count else f"{count} mask" + ("" if count == 1 else "s")


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

    def editor(self, parent: Any, values: FieldValues) -> Editor:
        """A read-out: masks are drawn and edited in a Mask Editor, never here."""
        del values
        widget = MaskList(parent)
        return Editor(
            widget,
            get=widget.value,
            set=widget.set_value,
            connect=lambda _cb: None,
            summary=widget.summary,
            spec=self,
        )


MASKS = declare(
    "masks",
    MasksField("Masks", "The masks drawn in a Mask Editor, which is where they are edited"),
    (),
)
