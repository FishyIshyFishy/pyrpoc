"""Channels composited into one colour image."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core.library import Library
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.plugins.program_io.outputs import ImageOutput
from pyrpoc.utils.widgets.colors import color_for_index, color_map_from_rgb
from pyrpoc.utils.widgets.data_panel import DataPanel, data_panel_registry
from pyrpoc.utils.widgets.frame_slider import FrameSlider
from pyrpoc.utils.widgets.levels import (
    LevelsControl,
    LevelsState,
    autoscale_levels,
    parse_levels,
)
from pyrpoc.utils.widgets.stage_link import StageLink


@data_panel_registry.register("overlay")
class OverlayPanel(DataPanel[ImageOutput]):
    display_name = "Overlaid"
    group = "Images"
    order = 20
    renders = ImageOutput

    def __init__(self, library: Library, inputs: ProgramInputs, stage: StageLink):
        super().__init__(library, inputs, stage)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._controls: list[LevelsControl] = []
        self._pending_state: list[LevelsState] = []

        root = QVBoxLayout(self.body)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)
        splitter = QSplitter(Qt.Orientation.Horizontal, self.body)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)
        splitter.addWidget(self.build_plot(splitter))
        splitter.addWidget(self.build_side(splitter))
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 300])
        self.frames = FrameSlider(self.body)
        self.frames.moved.connect(self.refresh)
        root.addWidget(self.frames)
        self.connect_source()

    def build_plot(self, parent: QWidget) -> pg.PlotWidget:
        plot = pg.PlotWidget(parent)
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setAspectLocked(True)
        plot.invertY(True)
        self._overlay_item = pg.ImageItem()
        self._overlay_item.setLevels((0.0, 1.0))
        plot.addItem(self._overlay_item)
        return plot

    def build_side(self, parent: QWidget) -> QScrollArea:
        scroll = QScrollArea(parent)
        scroll.setWidgetResizable(True)
        self._side_content = QWidget(scroll)
        self._side_layout = QHBoxLayout(self._side_content)
        self._side_layout.setContentsMargins(0, 0, 0, 0)
        self._side_layout.setSpacing(8)
        self._side_layout.addStretch(1)
        scroll.setWidget(self._side_content)
        scroll.setMinimumWidth(180)
        return scroll

    def frame(self) -> np.ndarray | None:
        """The image the frame slider is on, if there is one."""
        shown = self.shown_output
        return None if shown is None or not len(shown) else shown[self.frames.index()]

    def draw(self, program_output: ImageOutput) -> None:
        """Composite the image the frame slider is on, which follows the newest while on the
        last."""
        if not len(program_output):
            return
        self.frames.take_count(len(program_output))
        arr = program_output[self.frames.index()]
        self.sync_controls(arr.shape[0])
        for index in range(arr.shape[0]):
            self.update_channel(index, arr[index])
        self.update_overlay()

    def clear(self) -> None:
        self._overlay_item.setImage(
            np.zeros((1, 1, 3), dtype=np.float32), autoLevels=False, levels=(0.0, 1.0)
        )
        self.frames.reset()
        self.sync_controls(0)

    def sync_controls(self, count: int) -> None:
        """Add or drop controls to match ``count``, leaving the levels of controls that
        stay."""
        if len(self._controls) == count:
            return
        while len(self._controls) > count:
            control = self._controls.pop()
            # Hidden, not unparented: a show the layout queued when it was added
            # would otherwise open it as a top-level window before deletion.
            control.hide()
            control.deleteLater()
        while len(self._controls) < count:
            self._controls.append(self.build_control(len(self._controls)))
        self.apply_pending_state()
        self.reflow_controls()

    def build_control(self, index: int) -> LevelsControl:
        # The source image is never drawn: it holds the channel's plane so the histogram has data.
        cmap = color_map_from_rgb(color_for_index(index))
        control = LevelsControl(self._side_content, pg.ImageItem(), cmap)
        control.header.addWidget(QLabel(f"Ch {index + 1}", control))
        control.changed.connect(self.update_overlay)
        return control

    def update_channel(self, index: int, channel: np.ndarray) -> None:
        self._controls[index].take(channel, autoscale_levels(channel))

    def update_overlay(self) -> None:
        arr = self.frame()
        if arr is None:
            return
        rgb = np.zeros((arr.shape[1], arr.shape[2], 3), dtype=np.float32)
        for channel, control in zip(arr, self._controls, strict=False):
            scaled = control.scaled(channel)
            lut = control.lookup_table()
            rgb += lut[np.rint(scaled * (len(lut) - 1)).astype(np.intp)]
        self._overlay_item.setImage(np.clip(rgb, 0.0, 1.0), autoLevels=False, levels=(0.0, 1.0))

    def reflow_controls(self) -> None:
        while (item := self._side_layout.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.setParent(self._side_content)
        for control in self._controls:
            self._side_layout.addWidget(control)
        self._side_layout.addStretch(1)

    def export_persistence_state(self) -> dict[str, Any]:
        return {"channels": [control.export_state().to_json() for control in self._controls]}

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        # Applied once controls exist, since none do yet.
        self._pending_state = parse_levels(state.get("channels"))
        self.apply_pending_state()

    def apply_pending_state(self) -> None:
        """Apply saved settings, once."""
        # Once, so live edits during a run are not overwritten on every frame.
        if not self._pending_state or not self._controls:
            return
        for control, saved in zip(self._controls, self._pending_state, strict=False):
            control.restore(saved)
        self.update_overlay()
        self._pending_state = []
