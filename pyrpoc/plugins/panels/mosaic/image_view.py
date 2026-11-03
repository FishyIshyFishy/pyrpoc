"""The colour image both of the mosaic panel's views show."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget


class ColourImage(pg.PlotWidget):
    """An aspect-locked RGB image, already scaled to 0..1 by the channel strip."""

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

    def clear_image(self) -> None:
        self.item.clear()
