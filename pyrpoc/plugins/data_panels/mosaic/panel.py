"""A tiled acquisition, as a stack of frames or stitched into one mosaic.

The stack works for any ``Image2D``. The mosaic needs the layout its program
wrote into the output's metadata; without one, the Mosaic button is disabled.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
)

from pyrpoc.structs.data_library.data import Image2D
from pyrpoc.structs.data_library.dataset import Dataset
from pyrpoc.structs.data_library.library import Library
from pyrpoc.structs.plugins.data_panels import DataPanel, data_panel_registry

from .layout import MosaicLayout, read_layout
from .mosaic_view import MosaicView
from .stack_view import StackView

NO_LAYOUT_TIP = "This dataset has no mosaic layout; it was not acquired as a mosaic"


@data_panel_registry.register("mosaic")
class MosaicPanel(DataPanel):
    display_name = "Mosaic"
    renders = [Image2D]

    def __init__(self, library: Library):
        super().__init__(library)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.layout_found: MosaicLayout | None = None
        # What the user chose; the mosaic shows only when the dataset has a layout.
        self.wants_mosaic = False
        # A saved channel waits here until a dataset has that many channels.
        self._pending_channel: int | None = None

        root = QVBoxLayout(self.body)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self.build_toolbar())
        self.views = QStackedWidget(self.body)
        self.stack = StackView(self.views)
        self.mosaic = MosaicView(self.views)
        self.views.addWidget(self.stack)
        self.views.addWidget(self.mosaic)
        root.addWidget(self.views, 1)
        self.connect_source()

    def build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.stack_button = QPushButton("Stack", self.body)
        self.mosaic_button = QPushButton("Mosaic", self.body)
        self.modes = QButtonGroup(self.body)
        for button in (self.stack_button, self.mosaic_button):
            button.setCheckable(True)
            self.modes.addButton(button)
            bar.addWidget(button)
        self.stack_button.setChecked(True)
        self.modes.buttonClicked.connect(self.on_mode_clicked)

        bar.addSpacing(12)
        bar.addWidget(QLabel("Channel:", self.body))
        self.channel = QComboBox(self.body)
        self.channel.currentIndexChanged.connect(lambda _index: self.redraw())
        bar.addWidget(self.channel)
        self.autoscale = QCheckBox("Autoscale", self.body)
        self.autoscale.setChecked(True)
        self.autoscale.toggled.connect(self.on_autoscale_toggled)
        bar.addWidget(self.autoscale)
        bar.addStretch(1)
        return bar

    def show_frame(self, dataset: Dataset, frame: np.ndarray) -> None:
        self.sync_channels(dataset.resolved_channel_labels(frame.shape[0]))
        self.layout_found = self.read_layout(dataset)
        self.redraw()

    def read_layout(self, dataset: Dataset) -> MosaicLayout | None:
        """The dataset's layout, enabling the Mosaic button only when there is
        one. Metadata may come from a file, so an unreadable layout is shown
        on the button rather than raised."""
        try:
            layout = read_layout(dataset.metadata)
            tip = NO_LAYOUT_TIP if layout is None else "Stitch the tiles into one image"
        except ValueError as exc:
            layout, tip = None, f"The mosaic layout could not be read: {exc}"
        self.mosaic_button.setEnabled(layout is not None)
        self.mosaic_button.setToolTip(tip)
        return layout

    def sync_channels(self, labels: list[str]) -> None:
        """Rename or resize the channel list, keeping the chosen channel."""
        current = [self.channel.itemText(index) for index in range(self.channel.count())]
        if current == labels:
            return
        chosen = self.channel.currentIndex()
        if self._pending_channel is not None:
            chosen, self._pending_channel = self._pending_channel, None
        self.channel.blockSignals(True)
        self.channel.clear()
        self.channel.addItems(labels)
        self.channel.setCurrentIndex(min(max(chosen, 0), len(labels) - 1))
        self.channel.blockSignals(False)

    def redraw(self) -> None:
        """Draw the dataset in whichever view is showing; the hidden one waits."""
        dataset = self.shown_dataset
        if dataset is None or self.channel.currentIndex() < 0:
            return
        frames, channel, layout = dataset.frames(), self.channel.currentIndex(), self.layout_found
        if self.wants_mosaic and layout is not None:
            self.mosaic_button.setChecked(True)
            self.views.setCurrentWidget(self.mosaic)
            self.mosaic.show_frames(dataset.id, frames, layout, channel)
        else:
            self.stack_button.setChecked(True)
            self.views.setCurrentWidget(self.stack)
            self.stack.show_frames(frames, layout, channel)

    def on_mode_clicked(self, button: object) -> None:
        self.wants_mosaic = button is self.mosaic_button
        self.redraw()

    def on_autoscale_toggled(self, checked: bool) -> None:
        self.stack.image.autoscale = self.mosaic.image.autoscale = checked
        self.redraw()

    def clear(self) -> None:
        self.layout_found = None
        self.stack_button.setChecked(True)
        self.mosaic_button.setEnabled(False)
        self.mosaic_button.setToolTip(NO_LAYOUT_TIP)
        self.stack.clear()
        self.mosaic.clear()

    def export_persistence_state(self) -> dict[str, Any]:
        return {
            "mode": "mosaic" if self.wants_mosaic else "stack",
            "channel": self.channel.currentIndex(),
            "autoscale": self.autoscale.isChecked(),
            "fallback_overlap": self.mosaic.fallback.value(),
        }

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        """Parse saved settings from session JSON, a boundary: a malformed
        entry keeps its default."""
        self.wants_mosaic = state.get("mode") == "mosaic"
        channel = state.get("channel")
        if isinstance(channel, int) and channel >= 0:
            if self.channel.count():
                self.channel.setCurrentIndex(min(channel, self.channel.count() - 1))
            else:
                self._pending_channel = channel
        autoscale = state.get("autoscale")
        if isinstance(autoscale, bool):
            self.autoscale.setChecked(autoscale)
        overlap = state.get("fallback_overlap")
        if isinstance(overlap, (int, float)):
            self.mosaic.fallback.setValue(float(overlap))
        self.redraw()
