"""``ColourImage``: an aspect-locked RGB image, one plot unit per pixel unless it is placed."""

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

    def show_rgb(self, rgb: np.ndarray) -> None:
        self.item.setImage(rgb, autoLevels=False, levels=(0.0, 1.0))

    def place(self, rect: QRectF) -> None:
        """Spread the image over ``rect``, in whatever units the plot is in."""
        self.item.setRect(rect)

    def clear_image(self) -> None:
        self.item.clear()
