"""One curve per channel of a 1-D spectrum.

The x axis is the sample index: what a bin means in nm is the spectrometer's
calibration, not the shape contract.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout

from pyrpoc.qt_components.colors import color_for_index
from pyrpoc.structs.data_library.data import Spectrum1D
from pyrpoc.structs.data_library.dataset import Dataset
from pyrpoc.structs.data_library.library import Library
from pyrpoc.structs.plugins.data_panels import DataPanel, data_panel_registry


@data_panel_registry.register("spectrum")
class SpectrumPanel(DataPanel):
    display_name = "Spectrum"
    renders = [Spectrum1D]

    def __init__(self, library: Library):
        super().__init__(library)
        self._curves: list[pg.PlotDataItem] = []
        self._legend_labels: list[str] = []

        body_root = QVBoxLayout(self.body)
        body_root.setContentsMargins(6, 6, 6, 6)
        self._plot = pg.PlotWidget(self.body)
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setLabel("bottom", "Sample")
        self._plot.setLabel("left", "Counts")
        self._legend = self._plot.addLegend(offset=(-10, 10))
        body_root.addWidget(self._plot, 1)
        self.connect_source()

    def show_frame(self, dataset: Dataset, frame: np.ndarray) -> None:
        arr = np.asarray(frame, dtype=np.float32)
        self.sync_curves(arr.shape[0], dataset.resolved_channel_labels(arr.shape[0]))
        xs = np.arange(arr.shape[1], dtype=np.float32)
        for index, curve in enumerate(self._curves):
            curve.setData(xs, arr[index])

    def clear(self) -> None:
        self.sync_curves(0, [])

    def sync_curves(self, count: int, labels: list[str]) -> None:
        """One curve per channel. The legend is refilled rather than diffed: a
        stale row outlives its curve and would label the wrong trace. Same
        labels mean the same curves, so nothing is touched."""
        if labels == self._legend_labels and len(self._curves) == count:
            return
        self._legend_labels = list(labels)
        while len(self._curves) > count:
            self._plot.removeItem(self._curves.pop())
        while len(self._curves) < count:
            color = color_for_index(len(self._curves))
            self._curves.append(self._plot.plot(pen=pg.mkPen(color, width=2)))
        self._legend.clear()
        for curve, name in zip(self._curves, labels, strict=True):
            self._legend.addItem(curve, name)
