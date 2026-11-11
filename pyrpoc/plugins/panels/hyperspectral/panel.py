"""The Hyperspectral panel: a sweep's image at one delay, chosen on a slider, above its spectrum,
the mean of a region of each image against the delay it was taken at."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSplitter,
    QVBoxLayout,
)

from pyrpoc.core.library import Library
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.plugins.program_io.outputs import HyperspectralOutput
from pyrpoc.utils.widgets.data_panel import DataPanel, data_panel_registry
from pyrpoc.utils.widgets.levels import autoscale_levels

from .views import RegionImage, SpectrumPlot


@data_panel_registry.register("hyperspectral")
class HyperspectralPanel(DataPanel[HyperspectralOutput]):
    display_name = "Hyperspectral"
    group = "Spectra"
    order = 20
    renders = HyperspectralOutput

    def __init__(self, library: Library, inputs: ProgramInputs):
        super().__init__(library, inputs)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.delays: tuple[float, ...] = ()
        self.images: tuple[np.ndarray, ...] = ()
        # Restored before the first image says how many channels there are.
        self.wanted_channel = 0

        root = QVBoxLayout(self.body)
        root.setContentsMargins(6, 6, 6, 6)
        root.addLayout(self.build_controls())
        split = QSplitter(Qt.Orientation.Vertical, self.body)
        self.image = RegionImage(split)
        self.image.region_moved.connect(self.show_spectrum)
        split.addWidget(self.image)
        self.spectrum = SpectrumPlot(split)
        self.spectrum.picked.connect(self.slider_to)
        split.addWidget(self.spectrum)
        root.addWidget(split, 1)
        root.addLayout(self.build_slider_row())
        self.connect_source()

    def build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Channel:", self.body))
        self.channel_box = QComboBox(self.body)
        self.channel_box.currentIndexChanged.connect(lambda _index: self.redraw())
        row.addWidget(self.channel_box, 1)
        self.region_box = QCheckBox("Region", self.body)
        self.region_box.setToolTip("Take the spectrum over the box on the image, not all of it")
        self.region_box.setChecked(True)
        self.region_box.toggled.connect(self.on_region_toggled)
        row.addWidget(self.region_box)
        return row

    def build_slider_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal, self.body)
        # Qt's default range is 0-99; at 0-0 the first sweep shown starts on its newest image.
        self.slider.setMaximum(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(lambda _value: self.show_step())
        self.position = QLabel("", self.body)
        row.addWidget(self.slider, 1)
        row.addWidget(self.position)
        return row

    @property
    def channel(self) -> int:
        return max(self.channel_box.currentIndex(), 0)

    def draw(self, program_output: HyperspectralOutput) -> None:
        delays, images = program_output.taken()
        if not images:
            return
        self.delays, self.images = delays, images
        self.sync_channels(program_output.channels)
        self.sync_slider(len(images))
        self.redraw()

    def redraw(self) -> None:
        if self.images:
            self.show_step()
            self.show_spectrum()

    def sync_channels(self, labels: tuple[str, ...]) -> None:
        """List the channels, keeping the one chosen."""
        if [self.channel_box.itemText(i) for i in range(self.channel_box.count())] == list(labels):
            return
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        self.channel_box.addItems(labels)
        self.channel_box.setCurrentIndex(min(self.wanted_channel, len(labels) - 1))
        self.channel_box.blockSignals(False)

    def sync_slider(self, count: int) -> None:
        """Take the images so far, following a new last one only if the slider was on the last."""
        following = self.slider.value() >= self.slider.maximum()
        self.slider.blockSignals(True)
        self.slider.setMaximum(count - 1)
        self.slider.setEnabled(count > 1)
        if following:
            self.slider.setValue(count - 1)
        self.slider.blockSignals(False)

    def slider_to(self, index: int) -> None:
        self.slider.setValue(index)

    def show_step(self) -> None:
        """The image under the slider, levelled over the whole sweep so the spectrum's rise and
        fall shows as brightness."""
        if not self.images:
            return
        index = self.slider.value()
        planes = [image[self.channel] for image in self.images]
        low = min(autoscale_levels(plane)[0] for plane in planes)
        high = max(autoscale_levels(plane)[1] for plane in planes)
        self.image.show_plane(planes[index], (low, high))
        self.spectrum.mark(self.delays[index])
        self.position.setText(f"{index + 1}/{len(self.images)} · {self.delays[index]:.1f} µm")

    def show_spectrum(self) -> None:
        """Each image's mean over the region, or over all of it with the region off."""
        if not self.images:
            return
        _, height, width = self.images[0].shape
        rows, cols = (
            self.image.region_slices(height, width)
            if self.region_box.isChecked()
            else (slice(None), slice(None))
        )
        means = np.array([float(image[self.channel][rows, cols].mean()) for image in self.images])
        self.spectrum.show_spectrum(self.delays, means, self.channel)

    def on_region_toggled(self, checked: bool) -> None:
        self.image.region.setVisible(checked)
        self.show_spectrum()
        self.state_changed.emit()

    def clear(self) -> None:
        self.delays, self.images = (), ()
        self.slider.blockSignals(True)
        self.slider.setMaximum(0)
        self.slider.setEnabled(False)
        self.slider.blockSignals(False)
        self.position.setText("")
        self.image.clear_plane()
        self.spectrum.clear_spectrum()

    def export_persistence_state(self) -> dict[str, Any]:
        return {"channel": self.channel, "region": self.region_box.isChecked()}

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        self.wanted_channel = int(state.get("channel", 0))
        if self.wanted_channel < self.channel_box.count():
            self.channel_box.setCurrentIndex(self.wanted_channel)
        self.region_box.setChecked(bool(state.get("region", True)))
