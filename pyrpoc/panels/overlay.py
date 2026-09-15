"""Channels composited into one colour image.

Was ``displays/multichan_overlay_display.py``. Same compositing; the pixels now
come from a bound dataset rather than an array the widget owns.
"""

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

from pyrpoc.core.streams import Image2D

from .base import View
from .registry import view_registry


def color_for_index(index: int) -> tuple[int, int, int]:
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


@view_registry.register("overlay")
class OverlayView(View):
    display_name = "2D Overlaid"
    renders = [Image2D]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._controls: list[ChannelControl] = []
        self._pending_channel_state: list[dict[str, Any]] = []
        self._suspend_lut_signal = False

        root = QHBoxLayout(self.body)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal, self.body)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        self._plot = pg.PlotWidget(splitter)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.setAspectLocked(True)
        self._plot.invertY(True)
        self._overlay_item = pg.ImageItem()
        self._overlay_item.setLevels((0.0, 1.0))
        self._plot.addItem(self._overlay_item)
        splitter.addWidget(self._plot)

        self._side_scroll = QScrollArea(splitter)
        self._side_scroll.setWidgetResizable(True)
        self._side_content = QWidget(self._side_scroll)
        self._side_layout = QHBoxLayout(self._side_content)
        self._side_layout.setContentsMargins(0, 0, 0, 0)
        self._side_layout.setSpacing(8)
        self._side_layout.addStretch(1)
        self._side_scroll.setWidget(self._side_content)
        self._side_scroll.setMinimumWidth(180)
        splitter.addWidget(self._side_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 300])

    # -- rendering ------------------------------------------------------------ #

    def frame(self) -> np.ndarray | None:
        dataset = self.dataset()
        latest = dataset.latest() if dataset is not None else None
        return None if latest is None else np.asarray(latest, dtype=np.float32)

    def refresh(self) -> None:
        arr = self.frame()
        if arr is None:
            self.clear()
            return
        self.sync_controls(arr.shape[0])
        for index in range(arr.shape[0]):
            self.update_channel(index, arr[index])
        self.update_overlay()

    def clear(self) -> None:
        self._overlay_item.setImage(
            np.zeros((1, 1, 3), dtype=np.float32), autoLevels=False, levels=(0.0, 1.0)
        )
        self.sync_controls(0)

    # -- controls -------------------------------------------------------------- #

    def sync_controls(self, count: int) -> None:
        while len(self._controls) > count:
            control = self._controls.pop()
            control.root.setParent(None)
            control.root.deleteLater()

        while len(self._controls) < count:
            self._controls.append(self.build_control(len(self._controls)))

        self.apply_pending_channel_state()
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

        source_item = pg.ImageItem()
        rgb = color_for_index(index)
        cmap = color_map_from_rgb(rgb)
        source_item.setColorMap(cmap)
        hist_widget = pg.HistogramLUTWidget(root)
        hist_widget.setImageItem(source_item)
        hist_widget.item.gradient.setColorMap(cmap)
        layout.addWidget(hist_widget, 1)

        control = ChannelControl(
            root=root,
            autoscale_box=autoscale_box,
            hist_widget=hist_widget,
            source_item=source_item,
            rgb=rgb,
        )
        autoscale_box.toggled.connect(lambda _checked, i=index: self.on_channel_control_changed(i))
        hist_widget.item.sigLevelsChanged.connect(
            lambda _item, i=index: self.on_lut_levels_changed(i)
        )
        return control

    def on_channel_control_changed(self, index: int) -> None:
        arr = self.frame()
        if arr is None or index >= arr.shape[0]:
            return
        self.update_channel(index, arr[index])
        self.update_overlay()

    def on_lut_levels_changed(self, index: int) -> None:
        if self._suspend_lut_signal:
            return
        if index < 0 or index >= len(self._controls):
            return
        control = self._controls[index]
        min_val, max_val = control.hist_widget.item.getLevels()
        if max_val <= min_val:  # pyright:ignore
            max_val = min_val + 1e-12  # pyright:ignore
            control.hist_widget.item.setLevels(min_val, max_val)
        control.min_val = float(min_val)  # pyright:ignore
        control.max_val = float(max_val)  # pyright:ignore
        self.update_overlay()

    def update_channel(self, index: int, channel: np.ndarray) -> None:
        control = self._controls[index]
        control.source_item.setImage(channel, autoLevels=False)

        if control.autoscale_box.isChecked():
            min_val = float(np.min(channel))
            max_val = float(np.max(channel))
            if max_val <= min_val:
                max_val = min_val + 1e-12
            self.apply_levels(control, min_val, max_val)
        else:
            min_val, max_val = control.hist_widget.item.getLevels()
            self.apply_levels(control, float(min_val), float(max_val))  # pyright:ignore

    def apply_levels(self, control: ChannelControl, min_val: float, max_val: float) -> None:
        control.min_val = float(min_val)
        control.max_val = float(max_val)
        self._suspend_lut_signal = True
        try:
            control.source_item.setLevels((control.min_val, control.max_val))
            control.hist_widget.item.setLevels(control.min_val, control.max_val)
        finally:
            self._suspend_lut_signal = False

    def update_overlay(self) -> None:
        arr = self.frame()
        if arr is None:
            return
        height, width = int(arr.shape[1]), int(arr.shape[2])
        rgb = np.zeros((height, width, 3), dtype=np.float32)

        for index, control in enumerate(self._controls):
            if index >= arr.shape[0]:
                break
            lo = float(control.min_val)
            hi = float(control.max_val)
            if hi <= lo:
                hi = lo + 1e-12
            scaled = np.clip((arr[index] - lo) / (hi - lo), 0.0, 1.0)
            cr, cg, cb = control.rgb
            rgb[..., 0] += scaled * (cr / 255.0)
            rgb[..., 1] += scaled * (cg / 255.0)
            rgb[..., 2] += scaled * (cb / 255.0)

        self._overlay_item.setImage(np.clip(rgb, 0.0, 1.0), autoLevels=False, levels=(0.0, 1.0))

    def reflow_controls(self) -> None:
        for index in reversed(range(self._side_layout.count())):
            item = self._side_layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is not None:
                self._side_layout.removeWidget(widget)

        for control in self._controls:
            self._side_layout.addWidget(control.root)
        self._side_layout.addStretch(1)

    # -- persistence ------------------------------------------------------------ #

    def export_persistence_state(self) -> dict[str, Any]:
        return {
            "channels": [
                {
                    "index": index,
                    "autoscale": bool(control.autoscale_box.isChecked()),
                    "min_val": float(control.min_val),
                    "max_val": float(control.max_val),
                }
                for index, control in enumerate(self._controls)
            ]
        }

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        channels = state.get("channels", [])
        if not isinstance(channels, list):
            return
        parsed: list[dict[str, Any]] = []
        for row in channels:
            if not isinstance(row, dict):
                continue
            parsed.append(
                {
                    "index": int(row.get("index", len(parsed))),
                    "autoscale": bool(row.get("autoscale", True)),
                    "min_val": float(row.get("min_val", 0.0)),
                    "max_val": float(row.get("max_val", 1.0)),
                }
            )
        self._pending_channel_state = parsed
        self.apply_pending_channel_state()

    def apply_pending_channel_state(self) -> None:
        if not self._pending_channel_state or not self._controls:
            return
        for row in self._pending_channel_state:
            index = int(row.get("index", -1))
            if index < 0 or index >= len(self._controls):
                continue
            control = self._controls[index]
            min_val = float(row.get("min_val", control.min_val))
            max_val = float(row.get("max_val", control.max_val))
            control.autoscale_box.blockSignals(True)
            control.autoscale_box.setChecked(bool(row.get("autoscale", True)))
            control.autoscale_box.blockSignals(False)
            if max_val <= min_val:
                max_val = min_val + 1e-12
            self.apply_levels(control, min_val, max_val)
        if self.frame() is not None:
            self.update_overlay()
        # Apply persisted state once; live user changes during acquisition
        # should not be overwritten each frame.
        self._pending_channel_state = []
