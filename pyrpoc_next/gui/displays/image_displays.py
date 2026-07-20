"""Intensity image displays: single-channel streamed, per-channel tiled, RGB overlay.

Ported from the original pyrpoc displays (pyqtgraph PlotWidget + ImageItem +
HistogramLUTWidget + per-channel controls). Only the data boundary changed: they now
consume parcels via the DisplaySink protocol (handle() on the GUI thread) instead of
AcquiredData. Session persistence is deferred (G2), so the persistence entry points are
omitted; the levels/autoscale/name controls are otherwise unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pyrpoc_next.gui.displays.base import DisplayWidget, display_registry
from pyrpoc_next.structs.keys import DisplayKey
from pyrpoc_next.structs.manifest import DisplayManifest
from pyrpoc_next.structs.parcels import ImageFrameParcel, Parcel, PartialImageParcel

pg.setConfigOptions(imageAxisOrder="row-major")

# Shared black -> white -> red colormap; the last stop marks saturated/clipped pixels.
intensity_lut = pg.ColorMap(
    pos=np.array([0.0, 0.999, 1.0], dtype=float),
    color=np.array([[0, 0, 0, 255], [255, 255, 255, 255], [255, 0, 0, 255]], dtype=np.ubyte),
)


@dataclass
class ImageTile:
    root: QWidget
    name_edit: QLineEdit
    autoscale_box: QCheckBox
    image_item: pg.ImageItem
    hist_widget: pg.HistogramLUTWidget
    min_val: float = 0.0
    max_val: float = 1.0


@display_registry.register
class StreamedDisplay(DisplayWidget):
    """Live-updating progressive single-channel image (first channel of the frame).

    Blank-region mitigation: rows with zero counts in a partial frame are filled from
    the previous complete frame so the unscanned portion shows history, not black.
    """

    manifest = DisplayManifest(
        key=DisplayKey.streamed, display_name="Single Channel Streamed",
        accepted_parcels=(ImageFrameParcel, PartialImageParcel),
    )

    def __init__(self):
        super().__init__()
        self.data_hw: np.ndarray | None = None
        self.last_complete_hw: np.ndarray | None = None
        self.suspend_lut_signal = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        self.tile = self.build_tile()
        outer.addWidget(self.tile.root)

    def handle(self, parcel: Parcel) -> None:
        data = np.asarray(parcel.data)
        if data.ndim == 3 and data.shape[0] >= 1:
            frame = data[0]
        elif data.ndim == 2:
            frame = data
        else:
            return
        frame = np.asarray(frame, dtype=np.float32)
        if isinstance(parcel, PartialImageParcel):
            self.update_partial(frame)
        else:
            self.update_final(frame)

    def clear(self) -> None:
        self.data_hw = None
        self.last_complete_hw = None
        self.tile.image_item.setImage(np.zeros((1, 1), dtype=np.float32), autoLevels=False)

    def get_channel_names(self) -> list[str]:
        return [self.tile.name_edit.text().strip() or "Intensity"]

    def get_normalized_data_3d(self) -> np.ndarray | None:
        """Float32 [1, H, W] normalized to [0, 1] — what the mask editor consumes."""
        if self.data_hw is None:
            return None
        arr = self.data_hw
        lo, hi = float(np.min(arr)), float(np.max(arr))
        norm = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr, dtype=np.float32)
        return np.clip(norm, 0.0, 1.0)[np.newaxis]

    def update_partial(self, frame: np.ndarray) -> None:
        if self.last_complete_hw is not None and self.last_complete_hw.shape == frame.shape:
            blended = frame.copy()
            empty_rows = frame.sum(axis=1) == 0
            blended[empty_rows] = self.last_complete_hw[empty_rows]
        else:
            blended = frame
        self.data_hw = blended
        self.tile.image_item.setImage(blended, autoLevels=False)
        if self.tile.autoscale_box.isChecked():
            hi = float(np.max(blended))
            if hi > self.tile.max_val:
                self.apply_levels(self.tile, 0.0, hi)

    def update_final(self, frame: np.ndarray) -> None:
        self.data_hw = frame
        self.last_complete_hw = frame
        self.tile.image_item.setImage(frame, autoLevels=False)
        if self.tile.autoscale_box.isChecked():
            lo, hi = float(np.min(frame)), float(np.max(frame))
            if hi <= lo:
                hi = lo + 1e-12
            self.apply_levels(self.tile, lo, hi)
        else:
            lo, hi = cast(tuple[float, float], self.tile.hist_widget.item.getLevels())
            self.apply_levels(self.tile, lo, hi)

    def build_tile(self) -> ImageTile:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)

        name_edit = QLineEdit(root)
        name_edit.setText("Intensity")
        name_edit.textChanged.connect(lambda *_: self.request_persist())
        root_layout.addWidget(name_edit)

        body = QHBoxLayout()
        root_layout.addLayout(body, 1)

        plot = pg.PlotWidget(root)
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setAspectLocked(True)
        plot.invertY(True)
        image_item = pg.ImageItem()
        image_item.setColorMap(intensity_lut)
        plot.addItem(image_item)
        body.addWidget(plot, 1)

        hist_widget = pg.HistogramLUTWidget(root)
        hist_widget.setImageItem(image_item)
        hist_widget.item.gradient.setColorMap(intensity_lut)

        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(4)
        right_col.addWidget(hist_widget, 1)
        autoscale_box = QCheckBox("Autoscale", root)
        autoscale_box.setChecked(True)
        right_col.addWidget(autoscale_box)
        body.addLayout(right_col)

        tile = ImageTile(
            root=root, name_edit=name_edit, autoscale_box=autoscale_box,
            image_item=image_item, hist_widget=hist_widget,
        )
        autoscale_box.toggled.connect(self.on_autoscale_toggled)
        hist_widget.item.sigLevelsChanged.connect(self.on_lut_levels_changed)
        return tile

    def on_autoscale_toggled(self, checked: bool) -> None:
        if self.data_hw is None:
            return
        if checked:
            lo, hi = float(np.min(self.data_hw)), float(np.max(self.data_hw))
            if hi <= lo:
                hi = lo + 1e-12
            self.apply_levels(self.tile, lo, hi)
        self.request_persist()

    def on_lut_levels_changed(self) -> None:
        if self.suspend_lut_signal:
            return
        lo, hi = cast(tuple[float, float], self.tile.hist_widget.item.getLevels())
        if hi <= lo:
            hi = lo + 1e-12
            self.tile.hist_widget.item.setLevels(lo, hi)
        self.tile.min_val, self.tile.max_val = lo, hi
        self.request_persist()

    def apply_levels(self, tile: ImageTile, lo: float, hi: float) -> None:
        tile.min_val, tile.max_val = lo, hi
        tile.image_item.setLevels((lo, hi))
        self.suspend_lut_signal = True
        try:
            tile.hist_widget.item.setLevels(lo, hi)
        finally:
            self.suspend_lut_signal = False


@dataclass
class ChannelTile:
    root: QWidget
    name_edit: QLineEdit
    autoscale_box: QCheckBox
    image_item: pg.ImageItem
    hist_widget: pg.HistogramLUTWidget
    min_val: float = 0.0
    max_val: float = 1.0


@display_registry.register
class TiledDisplay(DisplayWidget):
    """One tile per channel in a 2-column grid; each has its own levels and autoscale."""

    manifest = DisplayManifest(
        key=DisplayKey.tiled, display_name="2D Tiled",
        accepted_parcels=(ImageFrameParcel, PartialImageParcel),
    )

    def __init__(self):
        super().__init__()
        self.data_chw: np.ndarray | None = None
        self.tiles: list[ChannelTile] = []
        self.suspend_lut_signal = False

        outer = QVBoxLayout(self)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll)
        self.content = QWidget(self.scroll)
        self.grid = QGridLayout(self.content)
        self.grid.setContentsMargins(8, 8, 8, 8)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(10)
        self.scroll.setWidget(self.content)

    def handle(self, parcel: Parcel) -> None:
        self.set_data(parcel.data)

    def clear(self) -> None:
        self.data_chw = None
        self.sync_channel_tiles(0)

    def set_data(self, data_chw: np.ndarray) -> None:
        arr = np.asarray(data_chw, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] <= 0:
            return
        self.data_chw = arr
        self.sync_channel_tiles(arr.shape[0])
        for idx in range(arr.shape[0]):
            self.update_channel_image(idx, arr[idx])

    def get_channel_names(self) -> list[str]:
        return [tile.name_edit.text().strip() or f"Input {i + 1}" for i, tile in enumerate(self.tiles)]

    def get_normalized_data_3d(self) -> np.ndarray | None:
        if self.data_chw is None:
            return None
        arr = np.asarray(self.data_chw, dtype=np.float32)
        norm = np.zeros_like(arr, dtype=np.float32)
        for idx in range(arr.shape[0]):
            channel = arr[idx]
            lo, hi = float(np.min(channel)), float(np.max(channel))
            if hi > lo:
                norm[idx] = (channel - lo) / (hi - lo)
        return np.clip(norm, 0.0, 1.0)

    def sync_channel_tiles(self, count: int) -> None:
        while len(self.tiles) > count:
            tile = self.tiles.pop()
            tile.root.setParent(None)
            tile.root.deleteLater()
        while len(self.tiles) < count:
            self.tiles.append(self.build_tile(len(self.tiles)))
        self.reflow_tiles()

    def build_tile(self, index: int) -> ChannelTile:
        root = QWidget(self.content)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)

        name_edit = QLineEdit(root)
        name_edit.setText(f"Input {index + 1}")
        name_edit.textChanged.connect(lambda *_: self.request_persist())
        root_layout.addWidget(name_edit)

        body = QHBoxLayout()
        root_layout.addLayout(body, 1)

        plot = pg.PlotWidget(root)
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setAspectLocked(True)
        plot.invertY(True)
        image_item = pg.ImageItem()
        image_item.setColorMap(intensity_lut)
        plot.addItem(image_item)
        body.addWidget(plot, 1)

        hist_widget = pg.HistogramLUTWidget(root)
        hist_widget.setImageItem(image_item)
        hist_widget.item.gradient.setColorMap(intensity_lut)
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(4)
        right_col.addWidget(hist_widget, 1)
        autoscale_box = QCheckBox("Autoscale", root)
        autoscale_box.setChecked(True)
        right_col.addWidget(autoscale_box)
        body.addLayout(right_col)

        tile = ChannelTile(
            root=root, name_edit=name_edit, autoscale_box=autoscale_box,
            image_item=image_item, hist_widget=hist_widget,
        )
        autoscale_box.toggled.connect(lambda checked, i=index: self.on_autoscale_toggled(i, checked))
        hist_widget.item.sigLevelsChanged.connect(lambda _item, i=index: self.on_lut_levels_changed(i))
        return tile

    def on_autoscale_toggled(self, idx: int, checked: bool) -> None:
        if self.data_chw is None or idx < 0 or idx >= self.data_chw.shape[0]:
            return
        self.update_channel_image(idx, self.data_chw[idx])
        self.request_persist()

    def on_lut_levels_changed(self, idx: int) -> None:
        if self.suspend_lut_signal or idx < 0 or idx >= len(self.tiles):
            return
        tile = self.tiles[idx]
        lo, hi = cast(tuple[float, float], tile.hist_widget.item.getLevels())
        if hi <= lo:
            hi = lo + 1e-12
            tile.hist_widget.item.setLevels(lo, hi)
        tile.min_val, tile.max_val = lo, hi
        self.request_persist()

    def update_channel_image(self, idx: int, channel: np.ndarray) -> None:
        tile = self.tiles[idx]
        tile.image_item.setImage(channel, autoLevels=False)
        if tile.autoscale_box.isChecked():
            lo, hi = float(np.min(channel)), float(np.max(channel))
            if hi <= lo:
                hi = lo + 1e-12
            self.apply_levels(tile, lo, hi)
        else:
            lo, hi = cast(tuple[float, float], tile.hist_widget.item.getLevels())
            self.apply_levels(tile, lo, hi)

    def apply_levels(self, tile: ChannelTile, lo: float, hi: float) -> None:
        tile.min_val, tile.max_val = float(lo), float(hi)
        tile.image_item.setLevels((tile.min_val, tile.max_val))
        self.suspend_lut_signal = True
        try:
            tile.hist_widget.item.setLevels(tile.min_val, tile.max_val)
        finally:
            self.suspend_lut_signal = False

    def reflow_tiles(self) -> None:
        while (item := self.grid.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.setParent(self.content)
        columns = 2
        for idx, tile in enumerate(self.tiles):
            row, col = idx // columns, idx % columns
            self.grid.addWidget(tile.root, row, col)
            self.grid.setRowStretch(row, 1)
            self.grid.setColumnStretch(col, 1)


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (255, 80, 80), (80, 220, 120), (70, 150, 255), (255, 200, 70),
        (190, 110, 255), (70, 230, 230), (255, 120, 210), (180, 180, 180),
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


@display_registry.register
class MultiChannelDisplay(DisplayWidget):
    """All channels composited into an additive RGB overlay, one color per channel."""

    manifest = DisplayManifest(
        key=DisplayKey.multichannel, display_name="2D Overlaid",
        accepted_parcels=(ImageFrameParcel, PartialImageParcel),
    )

    def __init__(self):
        super().__init__()
        self.data_chw: np.ndarray | None = None
        self.controls: list[ChannelControl] = []
        self.suspend_lut_signal = False

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        self.plot = pg.PlotWidget(splitter)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.overlay_item = pg.ImageItem()
        self.overlay_item.setLevels((0.0, 1.0))
        self.plot.addItem(self.overlay_item)
        splitter.addWidget(self.plot)

        self.side_scroll = QScrollArea(splitter)
        self.side_scroll.setWidgetResizable(True)
        self.side_content = QWidget(self.side_scroll)
        self.side_layout = QHBoxLayout(self.side_content)
        self.side_layout.setContentsMargins(0, 0, 0, 0)
        self.side_layout.setSpacing(8)
        self.side_layout.addStretch(1)
        self.side_scroll.setWidget(self.side_content)
        self.side_scroll.setMinimumWidth(180)
        splitter.addWidget(self.side_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 300])

    def handle(self, parcel: Parcel) -> None:
        self.set_data(parcel.data)

    def clear(self) -> None:
        self.data_chw = None
        self.overlay_item.setImage(np.zeros((1, 1, 3), dtype=np.float32), autoLevels=False, levels=(0.0, 1.0))
        self.sync_controls(0)

    def set_data(self, data_chw: np.ndarray) -> None:
        arr = np.asarray(data_chw, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] <= 0:
            return
        self.data_chw = arr
        self.sync_controls(arr.shape[0])
        for idx in range(arr.shape[0]):
            self.update_channel(idx, arr[idx])
        self.update_overlay()

    def get_channel_names(self) -> list[str]:
        return [f"Input {i + 1}" for i in range(len(self.controls))]

    def get_normalized_data_3d(self) -> np.ndarray | None:
        if self.data_chw is None:
            return None
        arr = np.asarray(self.data_chw, dtype=np.float32)
        norm = np.zeros_like(arr, dtype=np.float32)
        for idx in range(arr.shape[0]):
            channel = arr[idx]
            lo, hi = float(np.min(channel)), float(np.max(channel))
            if hi > lo:
                norm[idx] = (channel - lo) / (hi - lo)
        return np.clip(norm, 0.0, 1.0)

    def sync_controls(self, count: int) -> None:
        while len(self.controls) > count:
            ctl = self.controls.pop()
            ctl.root.setParent(None)
            ctl.root.deleteLater()
        while len(self.controls) < count:
            self.controls.append(self.build_control(len(self.controls)))
        self.reflow_controls()

    def build_control(self, index: int) -> ChannelControl:
        root = QWidget(self.side_content)
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

        ctl = ChannelControl(
            root=root, autoscale_box=autoscale_box, hist_widget=hist_widget,
            source_item=source_item, rgb=rgb,
        )
        autoscale_box.toggled.connect(lambda checked, i=index: self.on_autoscale_toggled(i, checked))
        hist_widget.item.sigLevelsChanged.connect(lambda _item, i=index: self.on_lut_levels_changed(i))
        return ctl

    def on_autoscale_toggled(self, idx: int, checked: bool) -> None:
        del checked
        if self.data_chw is None or idx < 0 or idx >= self.data_chw.shape[0]:
            return
        self.update_channel(idx, self.data_chw[idx])
        self.update_overlay()
        self.request_persist()

    def on_lut_levels_changed(self, idx: int) -> None:
        if self.suspend_lut_signal or idx < 0 or idx >= len(self.controls):
            return
        ctl = self.controls[idx]
        lo, hi = cast(tuple[float, float], ctl.hist_widget.item.getLevels())
        if hi <= lo:
            hi = lo + 1e-12
            ctl.hist_widget.item.setLevels(lo, hi)
        ctl.min_val, ctl.max_val = float(lo), float(hi)
        self.update_overlay()
        self.request_persist()

    def update_channel(self, idx: int, channel: np.ndarray) -> None:
        ctl = self.controls[idx]
        ctl.source_item.setImage(channel, autoLevels=False)
        if ctl.autoscale_box.isChecked():
            lo, hi = float(np.min(channel)), float(np.max(channel))
            if hi <= lo:
                hi = lo + 1e-12
            self.apply_levels(ctl, lo, hi)
        else:
            lo, hi = cast(tuple[float, float], ctl.hist_widget.item.getLevels())
            self.apply_levels(ctl, float(lo), float(hi))

    def apply_levels(self, ctl: ChannelControl, lo: float, hi: float) -> None:
        ctl.min_val, ctl.max_val = float(lo), float(hi)
        self.suspend_lut_signal = True
        try:
            ctl.source_item.setLevels((ctl.min_val, ctl.max_val))
            ctl.hist_widget.item.setLevels(ctl.min_val, ctl.max_val)
        finally:
            self.suspend_lut_signal = False

    def update_overlay(self) -> None:
        if self.data_chw is None:
            return
        arr = self.data_chw
        height, width = int(arr.shape[1]), int(arr.shape[2])
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        for idx, ctl in enumerate(self.controls):
            if idx >= arr.shape[0]:
                break
            lo, hi = float(ctl.min_val), float(ctl.max_val)
            if hi <= lo:
                hi = lo + 1e-12
            scaled = np.clip((arr[idx] - lo) / (hi - lo), 0.0, 1.0)
            cr, cg, cb = ctl.rgb
            rgb[..., 0] += scaled * (cr / 255.0)
            rgb[..., 1] += scaled * (cg / 255.0)
            rgb[..., 2] += scaled * (cb / 255.0)
        self.overlay_item.setImage(np.clip(rgb, 0.0, 1.0), autoLevels=False, levels=(0.0, 1.0))

    def reflow_controls(self) -> None:
        for i in reversed(range(self.side_layout.count())):
            widget = self.side_layout.itemAt(i).widget()
            if widget is not None:
                self.side_layout.removeWidget(widget)
        for ctl in self.controls:
            self.side_layout.addWidget(ctl.root)
        self.side_layout.addStretch(1)
