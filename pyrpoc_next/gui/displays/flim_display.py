"""FLIM display: photon-count image, a lifetime map, and the global decay curve.

The lifetime map is an intensity-weighted mean arrival time — a fast proxy. A
per-pixel exponential fit can replace it later without touching anything else.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout

from pyrpoc_next.gui.displays.base import DisplayWidget, display_registry
from pyrpoc_next.structs.keys import DisplayKey
from pyrpoc_next.structs.manifest import DisplayManifest
from pyrpoc_next.structs.parcels import HistogramCubeParcel


@display_registry.register
class FlimDisplay(DisplayWidget):
    """Lifetime imaging from per-pixel decay histograms."""

    manifest = DisplayManifest(
        key=DisplayKey.flim, display_name="FLIM Lifetime", accepted_parcels=(HistogramCubeParcel,)
    )

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.intensity = pg.ImageView()
        self.lifetime = pg.ImageView()
        self.decay = pg.PlotWidget(title="Global decay")
        layout.addWidget(self.intensity)
        layout.addWidget(self.lifetime)
        layout.addWidget(self.decay)

    def handle(self, parcel):
        cube = np.asarray(parcel.data, dtype=np.float32)
        total = cube.sum(axis=2)
        self.intensity.setImage(total.T, autoLevels=True)

        bins = np.arange(cube.shape[2], dtype=np.float32) * parcel.bin_width_ps
        weighted = (cube * bins).sum(axis=2)
        mean_time = np.divide(weighted, total, out=np.zeros_like(total), where=total > 0)
        self.lifetime.setImage(mean_time.T, autoLevels=True)

        self.decay.clear()
        self.decay.plot(bins, cube.sum(axis=(0, 1)))
