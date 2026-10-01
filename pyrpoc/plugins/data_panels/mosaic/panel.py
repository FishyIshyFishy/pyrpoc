"""A tiled acquisition, as a stack of frames or stitched into one mosaic,
with every channel overlaid in its own colour.

The stack works for any ``Image2D``. The mosaic needs the layout its program
wrote into the output's metadata; without one, the Mosaic button is disabled.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
)

from pyrpoc.structs.data_library.data import Image2D
from pyrpoc.structs.data_library.dataset import Dataset
from pyrpoc.structs.data_library.library import Library
from pyrpoc.structs.plugins.data_panels import DataPanel, data_panel_registry

from .channels import ChannelStrip
from .layout import MosaicLayout, read_layout
from .mosaic_view import MosaicView
from .stack_view import StackView

NO_LAYOUT_TIP = "This dataset has no mosaic layout; it was not acquired as a mosaic"


@data_panel_registry.register("mosaic")
class MosaicPanel(DataPanel):
    display_name = "Mosaic"
    group = "2D Image Views"
    order = 22
    renders = [Image2D]

    def __init__(self, library: Library):
        super().__init__(library)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.layout_found: MosaicLayout | None = None
        self.labels: list[str] = []
        # What the user chose; the mosaic shows only when the dataset has a layout.
        self.wants_mosaic = False

        root = QVBoxLayout(self.body)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self.build_modes())
        splitter = QSplitter(Qt.Orientation.Horizontal, self.body)
        splitter.setChildrenCollapsible(False)
        self.views = QStackedWidget(splitter)
        self.stack = StackView(self.views)
        self.mosaic = MosaicView(self.views)
        self.views.addWidget(self.stack)
        self.views.addWidget(self.mosaic)
        self.channels = ChannelStrip(splitter)
        splitter.addWidget(self.views)
        splitter.addWidget(self.channels)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([800, 300])
        root.addWidget(splitter, 1)

        self.stack.moved.connect(self.show_stack_frame)
        self.mosaic.fallback_changed.connect(self.redraw)
        self.channels.changed.connect(self.recolour)
        self.connect_source()

    def build_modes(self) -> QHBoxLayout:
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
        bar.addStretch(1)
        return bar

    def show_frame(self, dataset: Dataset, frame: np.ndarray) -> None:
        self.labels = dataset.resolved_channel_labels(frame.shape[0])
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

    def redraw(self) -> None:
        """Draw the dataset in whichever view applies; the hidden one waits."""
        dataset, layout = self.shown_dataset, self.layout_found
        if dataset is None:
            return
        frames = dataset.frames()
        if self.wants_mosaic and layout is not None:
            self.mosaic_button.setChecked(True)
            self.views.setCurrentWidget(self.mosaic)
            self.show_planes(self.mosaic.stitch(dataset.id, frames, layout))
        else:
            self.stack_button.setChecked(True)
            self.views.setCurrentWidget(self.stack)
            self.stack.take_frames(frames, layout)
            self.show_stack_frame()

    def show_stack_frame(self) -> None:
        self.show_planes(self.stack.current_frame())

    def show_planes(self, image: np.ndarray) -> None:
        """Hand a ``(C, H, W)`` image to the channel strip, then draw it in colour."""
        self.channels.set_planes(list(image), self.labels)
        self.recolour()

    def recolour(self) -> None:
        """Redraw the shown image with the channels' current levels and visibility."""
        if not self.channels.planes:
            return
        view = self.mosaic if self.views.currentWidget() is self.mosaic else self.stack
        view.image.show_rgb(self.channels.rgb())

    def on_mode_clicked(self, button: object) -> None:
        self.wants_mosaic = button is self.mosaic_button
        self.redraw()

    def clear(self) -> None:
        self.layout_found = None
        self.stack_button.setChecked(True)
        self.mosaic_button.setEnabled(False)
        self.mosaic_button.setToolTip(NO_LAYOUT_TIP)
        self.channels.clear_planes()
        self.stack.clear()
        self.mosaic.clear()

    def export_persistence_state(self) -> dict[str, Any]:
        return {
            "mode": "mosaic" if self.wants_mosaic else "stack",
            "fallback_overlap": self.mosaic.fallback.value(),
            "channels": self.channels.export_state(),
        }

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        """Parse saved settings from session JSON, a boundary: a malformed
        entry keeps its default."""
        self.wants_mosaic = state.get("mode") == "mosaic"
        overlap = state.get("fallback_overlap")
        if isinstance(overlap, (int, float)):
            self.mosaic.fallback.setValue(float(overlap))
        self.channels.import_state(state.get("channels"))
        self.redraw()
