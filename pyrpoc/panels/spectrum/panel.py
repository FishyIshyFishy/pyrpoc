"""One curve per channel of a 1-D spectrum.

The simplest panel in the folder, and deliberately so: a spectrum needs no LUT,
no autoscale checkbox and no per-channel tile, because the axes already carry
the numbers a histogram widget exists to recover for an image.

Pyqtgraph autoranges on the data, so a live continuous run at one point rescales
as the signal changes rather than clipping. The x axis is the sample index --
what a bin means in nm is a spectrometer calibration, which is a device property
and arrives with the device, not with the shape contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from pyrpoc.structs.data import Spectrum1D

from ..base import Panel
from ..components.colors import color_for_index
from ..components.source_picker import SourcePicker
from ..registry import panel_registry

if TYPE_CHECKING:  # pragma: no cover
    from pyrpoc.data.dataset import Dataset
    from pyrpoc.data.library import DataLibrary


@panel_registry.register("spectrum")
class SpectrumPanel(Panel):
    #: Declared but never emitted: this panel has no spatial meaning to
    #: report, but Application.add_panel connects to it on every panel the
    #: registry can produce, so it must exist.
    point_picked = pyqtSignal(str, int, int)

    display_name = "Spectrum"
    renders = [Spectrum1D]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self._curves: list[pg.PlotDataItem] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        self.source = SourcePicker(self.renders, self)
        self.source.changed.connect(self.refresh)
        root.addWidget(self.source)
        self.body = QWidget(self)
        root.addWidget(self.body, 1)

        body_root = QVBoxLayout(self.body)
        body_root.setContentsMargins(6, 6, 6, 6)

        self._plot = pg.PlotWidget(self.body)
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setLabel("bottom", "Sample")
        self._plot.setLabel("left", "Counts")
        self._legend = self._plot.addLegend(offset=(-10, 10))
        body_root.addWidget(self._plot, 1)

    # -- binding --------------------------------------------------------------- #

    def attach_library(self, library: "DataLibrary") -> None:
        self.source.attach_library(library)

    def dataset(self) -> "Dataset | None":
        return self.source.current()

    def set_picking(self, active: bool) -> None:
        """No spatial meaning to report; a no-op, not a missing method."""
        del active

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
