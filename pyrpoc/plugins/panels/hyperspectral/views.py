"""The Hyperspectral panel's two views: one image of the sweep with a region drawn on it, and the
spectrum, each image's mean against its delay."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget

from pyrpoc.utils.widgets.colors import color_for_index

GREY = pg.ColorMap(
    pos=np.array([0.0, 1.0], dtype=float),
    color=np.array([[0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte),
)
# A new image is given a region over its middle, this fraction of it on a side.
REGION_FRACTION = 0.5


class RegionImage(pg.PlotWidget):
    """One channel of one image, aspect-locked with y growing downward, and a box that picks the
    region the spectrum is taken over."""

    # The region was moved or resized.
    region_moved = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setAspectLocked(True)
        self.invertY(True)
        self.item = pg.ImageItem()
        self.item.setColorMap(GREY)
        self.addItem(self.item)
        self.region = pg.RectROI((0, 0), (1, 1), pen=pg.mkPen("y", width=2))
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(lambda _roi: self.region_moved.emit())
        self.addItem(self.region)
        # The shape the region was last placed for, so a new shape gets a fresh region.
        self.placed_for: tuple[int, int] | None = None

    def show_plane(self, plane: np.ndarray, levels: tuple[float, float]) -> None:
        """Show a ``(H, W)`` plane, placing the region over its middle if its shape is new."""
        self.item.setImage(plane, autoLevels=False, levels=levels)
        if plane.shape != self.placed_for:
            self.place_region(plane.shape[0], plane.shape[1])

    def place_region(self, height: int, width: int) -> None:
        self.placed_for = (height, width)
        size = (max(1.0, width * REGION_FRACTION), max(1.0, height * REGION_FRACTION))
        corner = ((width - size[0]) / 2, (height - size[1]) / 2)
        self.region.setPos(corner, finish=False)
        self.region.setSize(size)

    def region_slices(self, height: int, width: int) -> tuple[slice, slice]:
        """The rows and columns of a ``(height, width)`` plane under the region, at least one
        pixel of each."""
        x, y = (float(each) for each in self.region.pos())
        w, h = (float(each) for each in self.region.size())
        x0 = min(max(int(np.floor(x)), 0), width - 1)
        y0 = min(max(int(np.floor(y)), 0), height - 1)
        x1 = min(max(int(np.ceil(x + w)), x0 + 1), width)
        y1 = min(max(int(np.ceil(y + h)), y0 + 1), height)
        return slice(y0, y1), slice(x0, x1)

    def clear_plane(self) -> None:
        self.item.clear()
        self.placed_for = None


class SpectrumPlot(pg.PlotWidget):
    """Each image's mean against the delay it was taken at, with the image shown marked; a click
    picks the image nearest the delay clicked."""

    # The index of the image nearest a click.
    picked = pyqtSignal(int)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "Delay stage (µm)")
        self.setLabel("left", "Mean intensity")
        self.curve = self.plot(symbol="o", symbolSize=6)
        self.marker = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen((160, 160, 160), style=Qt.PenStyle.DashLine)
        )
        self.addItem(self.marker)
        self.delays: tuple[float, ...] = ()
        plot_item, scene = self.getPlotItem(), self.sceneObj
        if plot_item is None or scene is None:
            raise RuntimeError("a new PlotWidget has no PlotItem or scene")
        self.view_box = plot_item.getViewBox()
        scene.sigMouseClicked.connect(self.on_clicked)

    def show_spectrum(self, delays: tuple[float, ...], means: np.ndarray, channel: int) -> None:
        self.delays = delays
        color = color_for_index(channel)
        self.curve.setData(
            np.asarray(delays, dtype=np.float64),
            means,
            pen=pg.mkPen(color, width=2),
            symbolBrush=pg.mkBrush(color),
            symbolPen=None,
        )

    def mark(self, delay_um: float) -> None:
        self.marker.setValue(delay_um)

    def on_clicked(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self.delays:
            return
        x = self.view_box.mapSceneToView(event.scenePos()).x()
        nearest = int(np.argmin(np.abs(np.asarray(self.delays) - x)))
        self.picked.emit(nearest)
        event.accept()

    def clear_spectrum(self) -> None:
        self.delays = ()
        self.curve.setData([], [])
