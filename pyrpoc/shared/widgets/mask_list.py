"""``MaskList``: a read-only list of drawn masks, each a thumbnail beside its name and size."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from pyrpoc.core.mask import Mask

THUMBNAIL_PX = 40


def thumbnail(mask: Mask) -> QPixmap:
    """The mask in white on black, scaled to fit ``THUMBNAIL_PX``."""
    lit = np.ascontiguousarray(np.where(mask.array > 0, 255, 0).astype(np.uint8))
    height, width = lit.shape
    image = QImage(lit.tobytes(), width, height, width, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(image).scaled(
        THUMBNAIL_PX, THUMBNAIL_PX, Qt.AspectRatioMode.KeepAspectRatio
    )


class MaskList(QWidget):
    """Shows the masks it is given; they are drawn and edited in a Mask Editor."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.masks: tuple[Mask, ...] = ()
        self.rows = QVBoxLayout(self)
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.setSpacing(2)
        self.set_value(())

    def value(self) -> tuple[Mask, ...]:
        return self.masks

    def set_value(self, masks: tuple[Mask, ...]) -> None:
        self.masks = masks
        while (item := self.rows.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not masks:
            empty = QLabel("none; draw one in a Mask Editor", self)
            empty.setEnabled(False)
            self.rows.addWidget(empty)
        for mask in masks:
            self.rows.addWidget(self.build_row(mask))

    def build_row(self, mask: Mask) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        picture = QLabel(row)
        picture.setPixmap(thumbnail(mask))
        layout.addWidget(picture)
        layout.addWidget(QLabel(mask.describe(), row), 1)
        return row

    def summary(self) -> str:
        count = len(self.masks)
        return "none" if not count else f"{count} mask" + ("" if count == 1 else "s")
