"""``ColourImage``: an aspect-locked RGB image, placed in whatever units its caller chooses."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QWidget


class ColourImage(pg.PlotWidget):
    """An RGB image already scaled to 0..1, with y growing downward."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setAspectLocked(True)
        self.invertY(True)
        self.item = pg.ImageItem()
        self.addItem(self.item)

    def show_rgb(self, rgb: np.ndarray, rect: QRectF | None) -> None:
        """Show ``rgb``, spread over ``rect`` in plot units, or one unit per pixel."""
        self.item.setImage(rgb, autoLevels=False, levels=(0.0, 1.0))
        if rect is not None:
            self.item.setRect(rect)

    def clear_image(self) -> None:
        self.item.clear()
