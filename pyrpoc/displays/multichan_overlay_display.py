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

from pyrpoc.backend_utils.acquired_data import AcquiredData, DataKind
from pyrpoc.rpoc.types import RPOCImageInput
from .base_display import BaseDisplay
from .display_registry import display_registry


def _color_for_index(index: int) -> tuple[int, int, int]:
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


def _color_map_from_rgb(rgb: tuple[int, int, int]) -> pg.ColorMap:
    r, g, b = rgb
    return pg.ColorMap(
        pos=np.array([0.0, 1.0], dtype=float),
        color=np.array([[0, 0, 0, 255], [r, g, b, 255]], dtype=np.ubyte),
    )


@dataclass
class _ChannelControl:
    root: QWidget
    autoscale_box: QCheckBox
    hist_widget: pg.HistogramLUTWidget
    source_item: pg.ImageItem
    rgb: tuple[int, int, int]
    min_val: float = 0.0
    max_val: float = 1.0


@display_registry.register("multichan_overlay")
class MultiChannelOverlayDisplay(BaseDisplay):
    DISPLAY_KEY = "multichan_overlay"
    DISPLAY_NAME = "Multichannel Overlay Display"
    ACCEPTED_KINDS = [DataKind.INTENSITY_FRAME, DataKind.PARTIAL_FRAME]
    DISPLAY_PARAMETERS = {}

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._data_chw: np.ndarray | None = None
        self._controls: list[_ChannelControl] = []
        self._pending_channel_state: list[dict[str, Any]] = []
        self._suspend_lut_signal = False

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
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

    def configure(self, params: dict[str, Any]) -> None:
        del params

    def render(self, acquired: AcquiredData) -> None:
        self.set_data(acquired.data)

    def clear(self) -> None:
        self._data_chw = None
        self._overlay_item.setImage(np.zeros((1, 1, 3), dtype=np.float32), autoLevels=False, levels=(0.0, 1.0))
        self._sync_controls(0)

    def set_data(self, data_chw: np.ndarray) -> None:
        arr = np.asarray(data_chw, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError("MultiChannelOverlayDisplay expects channels-first data [C, H, W]")
        if arr.shape[0] <= 0:
            raise ValueError("MultiChannelOverlayDisplay requires at least one channel")
        if arr.shape[1] <= 0 or arr.shape[2] <= 0:
            raise ValueError("MultiChannelOverlayDisplay received invalid spatial dimensions")

        self._data_chw = arr
        self._sync_controls(arr.shape[0])
        for idx in range(arr.shape[0]):
            self._update_channel(idx, arr[idx], from_data=True)
        self._update_overlay()

    def get_channel_names(self) -> list[str]:
        return [f"Input {i + 1}" for i in range(len(self._controls))]

    def set_channel_names(self, names: list[str]) -> None:
        del names
        self.request_persist()

    def export_rpoc_input(self) -> RPOCImageInput | None:
        if self._data_chw is None:
            return None
        return RPOCImageInput(
            data=self._data_chw.astype(np.float32, copy=True),
            channel_labels=self.get_channel_names(),
            source_id=self.DISPLAY_KEY,
        )

    def get_normalized_data_3d(self) -> np.ndarray | None:
        if self._data_chw is None:
            return None
        arr = np.asarray(self._data_chw, dtype=np.float32)
        norm = np.zeros_like(arr, dtype=np.float32)
        for idx in range(arr.shape[0]):
            channel = arr[idx]
            lo = float(np.min(channel))
            hi = float(np.max(channel))
            if hi > lo:
                norm[idx] = (channel - lo) / (hi - lo)
        return np.clip(norm, 0.0, 1.0)

    def export_persistence_state(self) -> dict[str, Any]:
        channels: list[dict[str, Any]] = []
        for idx, ctl in enumerate(self._controls):
            channels.append(
                {
                    "index": idx,
                    "autoscale": bool(ctl.autoscale_box.isChecked()),
                    "min_val": float(ctl.min_val),
                    "max_val": float(ctl.max_val),
                }
            )
        return {"channels": channels}

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        channels = state.get("channels", [])
        if isinstance(channels, list):
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
            self._apply_pending_channel_state()

    def _sync_controls(self, count: int) -> None:
        while len(self._controls) > count:
            ctl = self._controls.pop()
            ctl.root.setParent(None)
            ctl.root.deleteLater()

        while len(self._controls) < count:
            idx = len(self._controls)
            self._controls.append(self._build_control(idx))

        self._apply_pending_channel_state()
        self._reflow_controls()

    def _build_control(self, index: int) -> _ChannelControl:
        root = QWidget(self._side_content)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        label = QLabel(f"Ch {index + 1}", root)
        layout.addWidget(label)

        autoscale_box = QCheckBox("Autoscale", root)
        autoscale_box.setChecked(True)
        layout.addWidget(autoscale_box)

        source_item = pg.ImageItem()
        rgb = _color_for_index(index)
        cmap = _color_map_from_rgb(rgb)
        source_item.setColorMap(cmap)
        hist_widget = pg.HistogramLUTWidget(root)
        hist_widget.setImageItem(source_item)
        hist_widget.item.gradient.setColorMap(cmap)
        layout.addWidget(hist_widget, 1)

        ctl = _ChannelControl(
            root=root,
            autoscale_box=autoscale_box,
            hist_widget=hist_widget,
            source_item=source_item,
            rgb=rgb,
        )
        autoscale_box.toggled.connect(lambda checked, i=index: self._on_autoscale_toggled(i, checked))
        hist_widget.item.sigLevelsChanged.connect(lambda _item, i=index: self._on_lut_levels_changed(i))
        return ctl

    def _on_autoscale_toggled(self, idx: int, checked: bool) -> None:
        if idx < 0 or idx >= len(self._controls):
            return
        if self._data_chw is None or idx >= self._data_chw.shape[0]:
            return
        if checked:
            self._update_channel(idx, self._data_chw[idx], from_data=True)
        else:
            self._update_channel(idx, self._data_chw[idx], from_data=False)
        self._update_overlay()
        self.request_persist()

    def _on_lut_levels_changed(self, idx: int) -> None:
        if self._suspend_lut_signal:
            return
        if idx < 0 or idx >= len(self._controls):
            return
        ctl = self._controls[idx]
        min_val, max_val = ctl.hist_widget.item.getLevels()
        if max_val <= min_val:#pyright:ignore
            max_val = min_val + 1e-12 #pyright:ignore
            ctl.hist_widget.item.setLevels(min_val, max_val)
        ctl.min_val = float(min_val)#pyright:ignore
        ctl.max_val = float(max_val)#pyright:ignore
        self._update_overlay()
        self.request_persist()

    def _update_channel(self, idx: int, channel: np.ndarray, from_data: bool) -> None:
        del from_data
        ctl = self._controls[idx]
        ctl.source_item.setImage(channel, autoLevels=False)

        if ctl.autoscale_box.isChecked():
            min_val = float(np.min(channel))
            max_val = float(np.max(channel))
            if max_val <= min_val:
                max_val = min_val + 1e-12
            self._apply_levels(ctl, min_val, max_val)
        else:
            min_val, max_val = ctl.hist_widget.item.getLevels()
            self._apply_levels(ctl, float(min_val), float(max_val)) #pyright:ignore

    def _apply_levels(self, ctl: _ChannelControl, min_val: float, max_val: float) -> None:
        ctl.min_val = float(min_val)
        ctl.max_val = float(max_val)
        self._suspend_lut_signal = True
        try:
            ctl.source_item.setLevels((ctl.min_val, ctl.max_val))
            ctl.hist_widget.item.setLevels(ctl.min_val, ctl.max_val)
        finally:
            self._suspend_lut_signal = False

    def _update_overlay(self) -> None:
        if self._data_chw is None:
            return
        arr = self._data_chw
        h, w = int(arr.shape[1]), int(arr.shape[2])
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        for idx, ctl in enumerate(self._controls):
            if idx >= arr.shape[0]:
                break
            lo = float(ctl.min_val)
            hi = float(ctl.max_val)
            if hi <= lo:
                hi = lo + 1e-12
            scaled = np.clip((arr[idx] - lo) / (hi - lo), 0.0, 1.0)
            cr, cg, cb = ctl.rgb
            rgb[..., 0] += scaled * (cr / 255.0)
            rgb[..., 1] += scaled * (cg / 255.0)
            rgb[..., 2] += scaled * (cb / 255.0)

        self._overlay_item.setImage(np.clip(rgb, 0.0, 1.0), autoLevels=False, levels=(0.0, 1.0))

    def _apply_pending_channel_state(self) -> None:
        if not self._pending_channel_state or not self._controls:
            return
        for row in self._pending_channel_state:
            idx = int(row.get("index", -1))
            if idx < 0 or idx >= len(self._controls):
                continue
            ctl = self._controls[idx]
            autoscale = bool(row.get("autoscale", True))
            min_val = float(row.get("min_val", ctl.min_val))
            max_val = float(row.get("max_val", ctl.max_val))
            ctl.autoscale_box.blockSignals(True)
            ctl.autoscale_box.setChecked(autoscale)
            ctl.autoscale_box.blockSignals(False)
            if max_val <= min_val:
                max_val = min_val + 1e-12
            self._apply_levels(ctl, min_val, max_val)
        if self._data_chw is not None:
            self._update_overlay()
        # Apply persisted state once; live user changes during acquisition
        # should not be overwritten each frame.
        self._pending_channel_state = []

    def _reflow_controls(self) -> None:
        for i in reversed(range(self._side_layout.count())):
            item = self._side_layout.itemAt(i)
            widget = item.widget() #pyright:ignore
            if widget is not None:
                self._side_layout.removeWidget(widget)

        for ctl in self._controls:
            self._side_layout.addWidget(ctl.root)
        self._side_layout.addStretch(1)
