"""Channels composited into one colour image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.src.structs.data import Image2D, Library

from ..components.colors import color_for_index
from ..components.dataset_panel import DatasetPanel, panel_registry
from ..components.levels import autoscale_levels, mono_levels


def color_map_from_rgb(rgb: tuple[int, int, int]) -> pg.ColorMap:
    r, g, b = rgb
    return pg.ColorMap(
        pos=np.array([0.0, 1.0], dtype=float),
        color=np.array([[0, 0, 0, 255], [r, g, b, 255]], dtype=np.ubyte),
    )


@dataclass
class ChannelControl:
    root: QWidget
    autoscale_box: QCheckBox
    hist_widget: pg.HistogramLUTWidget
    source_item: pg.ImageItem
    rgb: tuple[int, int, int]
    min_val: float = 0.0
    max_val: float = 1.0


@dataclass
class ControlState:
    """One channel's saved settings, parsed from the workspace."""

    index: int
    autoscale: bool
    min_val: float
    max_val: float


@panel_registry.register("overlay")
class OverlayPanel(DatasetPanel):
    display_name = "2D Overlaid"
    renders = [Image2D]

    def __init__(self, library: Library):
        super().__init__(library)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._controls: list[ChannelControl] = []
        self._pending_state: list[ControlState] = []
        self._suspend_lut_signal = False

        root = QHBoxLayout(self.body)
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
        latest = self.latest()
        return None if latest is None else np.asarray(latest, dtype=np.float32)

    def refresh(self) -> None:
        arr = self.frame()
        if arr is None:
            self._overlay_item.setImage(
                np.zeros((1, 1, 3), dtype=np.float32), autoLevels=False, levels=(0.0, 1.0)
            )
            self.sync_controls(0)
            return
        self.sync_controls(arr.shape[0])
        for index in range(arr.shape[0]):
            self.update_channel(index, arr[index])
        self.update_overlay()

    def sync_controls(self, count: int) -> None:
        while len(self._controls) > count:
            control = self._controls.pop()
            control.root.setParent(None)
            control.root.deleteLater()
        while len(self._controls) < count:
            self._controls.append(self.build_control(len(self._controls)))
        self.apply_pending_state()
        self.reflow_controls()

    def build_control(self, index: int) -> ChannelControl:
        root = QWidget(self._side_content)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(QLabel(f"Ch {index + 1}", root))
        autoscale_box = QCheckBox("Autoscale", root)
        autoscale_box.setChecked(True)
        layout.addWidget(autoscale_box)

        rgb = color_for_index(index)
        cmap = color_map_from_rgb(rgb)
        source_item = pg.ImageItem()
        source_item.setColorMap(cmap)
        hist_widget = pg.HistogramLUTWidget(root)
        hist_widget.setImageItem(source_item)
        hist_widget.item.gradient.setColorMap(cmap)
        layout.addWidget(hist_widget, 1)

        control = ChannelControl(root, autoscale_box, hist_widget, source_item, rgb)
        autoscale_box.toggled.connect(lambda _checked, i=index: self.on_channel_control_changed(i))
        hist_widget.item.sigLevelsChanged.connect(
            lambda _item, i=index: self.on_lut_levels_changed(i)
        )
        return control

    def on_channel_control_changed(self, index: int) -> None:
        arr = self.frame()
        if arr is None:
            return
        self.update_channel(index, arr[index])
        self.update_overlay()

    def on_lut_levels_changed(self, index: int) -> None:
        if self._suspend_lut_signal:
            return
        control = self._controls[index]
        min_val, max_val = mono_levels(control.hist_widget)
        if max_val <= min_val:
            max_val = min_val + 1e-12
            control.hist_widget.item.setLevels(min_val, max_val)
        control.min_val = min_val
        control.max_val = max_val
        self.update_overlay()

    def update_channel(self, index: int, channel: np.ndarray) -> None:
        control = self._controls[index]
        control.source_item.setImage(channel, autoLevels=False)
        if control.autoscale_box.isChecked():
            self.apply_levels(control, *autoscale_levels(channel))
        else:
            self.apply_levels(control, *mono_levels(control.hist_widget))

    def apply_levels(self, control: ChannelControl, min_val: float, max_val: float) -> None:
        control.min_val = min_val
        control.max_val = max_val
        self._suspend_lut_signal = True
        try:
            control.source_item.setLevels((min_val, max_val))
            control.hist_widget.item.setLevels(min_val, max_val)
        finally:
            self._suspend_lut_signal = False

    def update_overlay(self) -> None:
        arr = self.frame()
        if arr is None:
            return
        rgb = np.zeros((arr.shape[1], arr.shape[2], 3), dtype=np.float32)
        for channel, control in zip(arr, self._controls, strict=False):
            lo, hi = control.min_val, max(control.max_val, control.min_val + 1e-12)
            scaled = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
            for axis, component in enumerate(control.rgb):
                rgb[..., axis] += scaled * (component / 255.0)
        self._overlay_item.setImage(np.clip(rgb, 0.0, 1.0), autoLevels=False, levels=(0.0, 1.0))

    def reflow_controls(self) -> None:
        while (item := self._side_layout.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.setParent(self._side_content)
        for control in self._controls:
            self._side_layout.addWidget(control.root)
        self._side_layout.addStretch(1)

    def export_persistence_state(self) -> dict[str, Any]:
        return {
            "channels": [
                {
                    "index": index,
                    "autoscale": control.autoscale_box.isChecked(),
                    "min_val": control.min_val,
                    "max_val": control.max_val,
                }
                for index, control in enumerate(self._controls)
            ]
        }

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        """Parse saved settings from workspace JSON, a boundary: malformed rows
        are skipped. Applied once controls exist, since none do yet."""
        channels = state.get("channels", [])
        if not isinstance(channels, list):
            return
        self._pending_state = [
            ControlState(
                index=int(row.get("index", position)),
                autoscale=bool(row.get("autoscale", True)),
                min_val=float(row.get("min_val", 0.0)),
                max_val=float(row.get("max_val", 1.0)),
            )
            for position, row in enumerate(channels)
            if isinstance(row, dict)
        ]
        self.apply_pending_state()

    def apply_pending_state(self) -> None:
        """Apply saved settings once, so live edits during a run are not
        overwritten on every frame."""
        if not self._pending_state or not self._controls:
            return
        for saved in self._pending_state:
            if not 0 <= saved.index < len(self._controls):
                continue
            control = self._controls[saved.index]
            control.autoscale_box.blockSignals(True)
            control.autoscale_box.setChecked(saved.autoscale)
            control.autoscale_box.blockSignals(False)
            self.apply_levels(control, saved.min_val, max(saved.max_val, saved.min_val + 1e-12))
        self.update_overlay()
        self._pending_state = []
