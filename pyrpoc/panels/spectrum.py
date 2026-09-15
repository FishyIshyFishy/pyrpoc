"""One curve per channel of a 1-D spectrum.

The simplest view in the folder, and deliberately so: a spectrum needs no LUT,
no autoscale checkbox and no per-channel tile, because the axes already carry
the numbers a histogram widget exists to recover for an image.

Pyqtgraph autoranges on the data, so a live continuous run at one point rescales
as the signal changes rather than clipping. The x axis is the sample index --
what a bin means in nm is a spectrometer calibration, which is a device property
and arrives with the device, not with the shape contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from pyrpoc.core.streams import Spectrum1D

from .base import View
from .registry import view_registry


def color_for_index(index: int) -> tuple[int, int, int]:
    """Same palette as the overlay view, so a channel keeps its colour."""
    palette = [
        (255, 80, 80),
        (80, 220, 120),
        (70, 150, 255),
        (255, 200, 70),
        (190, 110, 255),
        (70, 230, 230),
        (255, 120, 210),
        (180, 180, 180),
    ]
    return palette[index % len(palette)]


@view_registry.register("spectrum")
class SpectrumView(View):
    display_name = "Spectrum"
    renders = [Spectrum1D]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self._curves: list[pg.PlotDataItem] = []

        root = QVBoxLayout(self.body)
        root.setContentsMargins(6, 6, 6, 6)

        self._plot = pg.PlotWidget(self.body)
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setLabel("bottom", "Sample")
        self._plot.setLabel("left", "Counts")
        self._legend = self._plot.addLegend(offset=(-10, 10))
        root.addWidget(self._plot, 1)

    # -- rendering ------------------------------------------------------------ #

    def refresh(self) -> None:
        dataset = self.dataset()
        latest = dataset.latest() if dataset is not None else None
        if latest is None:
            self.clear()
            return

        arr = np.asarray(latest, dtype=np.float32)
        labels = dataset.resolved_channel_labels(arr.shape[0]) if dataset is not None else []
        self.sync_curves(arr.shape[0], labels)
        xs = np.arange(arr.shape[1], dtype=np.float32)
        for index, curve in enumerate(self._curves):
            curve.setData(xs, arr[index])

    def clear(self) -> None:
        self.sync_curves(0, [])

    def sync_curves(self, count: int, labels: list[str]) -> None:
        """One curve per channel, rebuilding the legend when the count moves.

        The legend is cleared and refilled rather than diffed: it holds one row
        per curve and a stale row outlives the curve it named, which on a
        channel-count change would label the wrong trace.
        """
        while len(self._curves) > count:
            curve = self._curves.pop()
            self._plot.removeItem(curve)

        while len(self._curves) < count:
            index = len(self._curves)
            curve = self._plot.plot(pen=pg.mkPen(color_for_index(index), width=2))
            self._curves.append(curve)

        if self._legend is not None:
            self._legend.clear()
            for index, curve in enumerate(self._curves):
                name = labels[index] if index < len(labels) else f"channel_{index}"
                self._legend.addItem(curve, name)

    # -- persistence ----------------------------------------------------------- #

    def export_persistence_state(self) -> dict[str, Any]:
        """Nothing to keep.

        There is no per-channel state to restore -- no LUT, no levels, no name
        override -- and the view range is pyqtgraph's to decide from the data.
        """
        return {}
