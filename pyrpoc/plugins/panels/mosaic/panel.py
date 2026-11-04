"""A mosaic, as a stack of its images or stitched into one, with every
channel overlaid in its own colour."""

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

from pyrpoc.core.library import Library
from pyrpoc.core.panels import DataPanel, data_panel_registry
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.plugins.program_io.outputs import Mosaic

from .channels import ChannelStrip
from .layout import layout_of
from .mosaic_view import MosaicView
from .stack_view import StackView


@data_panel_registry.register("mosaic")
class MosaicPanel(DataPanel[Mosaic]):
    display_name = "Mosaic"
    group = "2D Image Views"
    order = 22
    renders = Mosaic

    def __init__(self, library: Library, inputs: ProgramInputs):
        super().__init__(library, inputs)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.labels: list[str] = []
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

    def show_value(self, program_output: ProgramOutput[Mosaic], value: Mosaic) -> None:
        del program_output
        if not value.images:
            self.clear()
            return
        self.labels = list(value.images[0].channels)
        self.redraw()

    def redraw(self) -> None:
        """Draw the mosaic in whichever view is chosen; the hidden one waits."""
        program_output, mosaic = self.shown_output, self.shown_value
        if program_output is None or mosaic is None or not mosaic.images:
            return
        frames = [image.array for image in mosaic.images]
        layout = layout_of(mosaic)
        if self.wants_mosaic:
            self.mosaic_button.setChecked(True)
            self.views.setCurrentWidget(self.mosaic)
            self.show_planes(self.mosaic.stitch(program_output.id, frames, layout))
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
        """Take saved settings from session JSON; a malformed entry keeps its default."""
        self.wants_mosaic = state.get("mode") == "mosaic"
        overlap = state.get("fallback_overlap")
        if isinstance(overlap, (int, float)):
            self.mosaic.fallback.setValue(float(overlap))
        self.channels.import_state(state.get("channels"))
        self.redraw()
