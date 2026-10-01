"""One image with its histogram, which both of the mosaic panel's views show."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from pyrpoc.qt_components.levels import SATURATION_LUT, autoscale_levels, mono_levels


class LevelledImage(QWidget):
    """An aspect-locked image and its histogram. With ``autoscale`` off, a new
    image keeps the levels the user set on the histogram."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.autoscale = True
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget(self)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.item = pg.ImageItem()
        self.item.setColorMap(SATURATION_LUT)
        self.plot.addItem(self.item)
        layout.addWidget(self.plot, 1)

        self.histogram = pg.HistogramLUTWidget(self)
        self.histogram.setImageItem(self.item)
        self.histogram.item.gradient.setColorMap(SATURATION_LUT)
        layout.addWidget(self.histogram)

    def show_plane(self, plane: np.ndarray) -> None:
        self.item.setImage(plane, autoLevels=False)
        low, high = autoscale_levels(plane) if self.autoscale else mono_levels(self.histogram)
        self.item.setLevels((low, high))
        self.histogram.item.setLevels(low, high)

    def clear_plane(self) -> None:
        self.item.clear()
