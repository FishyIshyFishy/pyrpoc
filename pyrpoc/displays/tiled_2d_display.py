from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.backend_utils.acquired_data import AcquiredData, DataKind
from pyrpoc.rpoc.types import RPOCImageInput
from .base_display import BaseDisplay
from .display_registry import display_registry


@dataclass
class _ChannelTile:
    root: QWidget
    name_edit: QLineEdit
    autoscale_box: QCheckBox
    image_item: pg.ImageItem
    hist_widget: pg.HistogramLUTWidget
    min_val: float = 0.0
    max_val: float = 1.0


@display_registry.register("tiled_2d")
class Tiled2DDisplay(BaseDisplay):
    DISPLAY_KEY = "tiled_2d"
    DISPLAY_NAME = "2D Tiled"
    ACCEPTED_KINDS = [DataKind.INTENSITY_FRAME, DataKind.PARTIAL_FRAME]
    DISPLAY_PARAMETERS = {}

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._data_chw: np.ndarray | None = None
        self._tiles: list[_ChannelTile] = []
        self._pending_channel_state: list[dict[str, Any]] = []
        self._suspend_lut_signal = False
        self._lut = pg.ColorMap(
            pos=np.array([0.0, 0.999, 1.0], dtype=float),
            color=np.array(
                [
                    [0, 0, 0, 255],
                    [255, 255, 255, 255],
                    [255, 0, 0, 255],
                ],
                dtype=np.ubyte,
            ),
        )

        outer = QVBoxLayout(self)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        outer.addWidget(self._scroll)

        self._content = QWidget(self._scroll)
        self._grid = QGridLayout(self._content)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(10)
        self._scroll.setWidget(self._content)

    def configure(self, params: dict[str, Any]) -> None:
        del params

    def render(self, acquired: AcquiredData) -> None:
        self.set_data(acquired.data)

    def clear(self) -> None:
        self._data_chw = None
        self._sync_channel_tiles(0)

    def set_data(self, data_chw: np.ndarray) -> None:
        arr = np.asarray(data_chw, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError("Tiled2DDisplay expects channels-first data [C, H, W]")
        if arr.shape[0] <= 0:
            raise ValueError("Tiled2DDisplay requires at least one channel")
        if arr.shape[1] <= 0 or arr.shape[2] <= 0:
            raise ValueError("Tiled2DDisplay received invalid spatial dimensions")

        self._data_chw = arr
        self._sync_channel_tiles(arr.shape[0])
        for idx in range(arr.shape[0]):
            self._update_channel_image(idx, arr[idx], from_data=True)

    def get_channel_names(self) -> list[str]:
        return [tile.name_edit.text().strip() or f"Input {i + 1}" for i, tile in enumerate(self._tiles)]

    def set_channel_names(self, names: list[str]) -> None:
        for i, tile in enumerate(self._tiles):
            if i < len(names) and str(names[i]).strip():
                tile.name_edit.setText(str(names[i]).strip())
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
        for idx, tile in enumerate(self._tiles):
            channels.append(
                {
                    "index": idx,
                    "name": tile.name_edit.text().strip(),
                    "autoscale": bool(tile.autoscale_box.isChecked()),
                    "min_val": float(tile.min_val),
                    "max_val": float(tile.max_val),
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
                        "name": str(row.get("name", "")).strip(),
                        "autoscale": bool(row.get("autoscale", True)),
                        "min_val": float(row.get("min_val", 0.0)),
                        "max_val": float(row.get("max_val", 1.0)),
                    }
                )
            self._pending_channel_state = parsed
            self._apply_pending_channel_state()

    def _sync_channel_tiles(self, count: int) -> None:
        while len(self._tiles) > count:
            tile = self._tiles.pop()
            tile.root.setParent(None)
            tile.root.deleteLater()

        while len(self._tiles) < count:
            idx = len(self._tiles)
            self._tiles.append(self._build_tile(idx))

        self._apply_pending_channel_state()
        self._reflow_tiles()

    def _build_tile(self, index: int) -> _ChannelTile:
        root = QWidget(self._content)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)

        name_edit = QLineEdit(root)
        name_edit.setText(f"Input {index + 1}")
        name_edit.textChanged.connect(lambda *_args: self.request_persist())
        root_layout.addWidget(name_edit)

        body = QHBoxLayout()
        root_layout.addLayout(body, 1)

        plot = pg.PlotWidget(root)
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setAspectLocked(True)
        plot.invertY(True)
        image_item = pg.ImageItem()
        image_item.setColorMap(self._lut)
        plot.addItem(image_item)
        body.addWidget(plot, 1)

        hist_widget = pg.HistogramLUTWidget(root)
        hist_widget.setImageItem(image_item)
        hist_widget.item.gradient.setColorMap(self._lut)
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(4)
        right_col.addWidget(hist_widget, 1)

        autoscale_box = QCheckBox("Autoscale", root)
        autoscale_box.setChecked(True)
        right_col.addWidget(autoscale_box)
        body.addLayout(right_col)

        tile = _ChannelTile(
            root=root,
            name_edit=name_edit,
            autoscale_box=autoscale_box,
            image_item=image_item,
            hist_widget=hist_widget,
        )
        autoscale_box.toggled.connect(lambda checked, i=index: self._on_autoscale_toggled(i, checked))
        hist_widget.item.sigLevelsChanged.connect(lambda _item, i=index: self._on_lut_levels_changed(i))
        return tile

    def _on_autoscale_toggled(self, idx: int, checked: bool) -> None:
        if idx < 0 or idx >= len(self._tiles):
            return
        if checked and self._data_chw is not None and idx < self._data_chw.shape[0]:
            self._update_channel_image(idx, self._data_chw[idx], from_data=True)
        elif self._data_chw is not None and idx < self._data_chw.shape[0]:
            tile = self._tiles[idx]
            tile.image_item.setImage(self._data_chw[idx], autoLevels=False)
        self.request_persist()

    def _on_lut_levels_changed(self, idx: int) -> None:
        if self._suspend_lut_signal:
            return
        if idx < 0 or idx >= len(self._tiles):
            return
        tile = self._tiles[idx]
        min_val, max_val = tile.hist_widget.item.getLevels()
        if max_val <= min_val:
            max_val = min_val + 1e-12
            tile.hist_widget.item.setLevels(min_val, max_val)
        tile.min_val = float(min_val)
        tile.max_val = float(max_val)
        self.request_persist()

    def _update_channel_image(self, idx: int, channel: np.ndarray, from_data: bool) -> None:
        tile = self._tiles[idx]
        tile.image_item.setImage(channel, autoLevels=False)

        if tile.autoscale_box.isChecked():
            min_val = float(np.min(channel))
            max_val = float(np.max(channel))
            if max_val <= min_val:
                max_val = min_val + 1e-12
            self._apply_levels(tile, min_val, max_val)
        else:
            min_val, max_val = tile.hist_widget.item.getLevels()
            self._apply_levels(tile, min_val, max_val)

    def _apply_levels(self, tile: _ChannelTile, min_val: float, max_val: float) -> None:
        tile.min_val = float(min_val)
        tile.max_val = float(max_val)
        tile.image_item.setLevels((tile.min_val, tile.max_val))
        self._suspend_lut_signal = True
        try:
            tile.hist_widget.item.setLevels(tile.min_val, tile.max_val)
        finally:
            self._suspend_lut_signal = False

    def _apply_pending_channel_state(self) -> None:
        if not self._pending_channel_state or not self._tiles:
            return
        for row in self._pending_channel_state:
            idx = int(row.get("index", -1))
            if idx < 0 or idx >= len(self._tiles):
                continue
            tile = self._tiles[idx]
            name = str(row.get("name", "")).strip()
            autoscale = bool(row.get("autoscale", True))
            min_val = float(row.get("min_val", tile.min_val))
            max_val = float(row.get("max_val", tile.max_val))
            tile.name_edit.blockSignals(True)
            tile.name_edit.setText(name or f"Input {idx + 1}")
            tile.name_edit.blockSignals(False)
            tile.autoscale_box.blockSignals(True)
            tile.autoscale_box.setChecked(autoscale)
            tile.autoscale_box.blockSignals(False)
            if max_val <= min_val:
                max_val = min_val + 1e-12
            self._apply_levels(tile, min_val, max_val)
        # Apply persisted state once; live user changes during acquisition
        # should not be overwritten each frame.
        self._pending_channel_state = []

    def _reflow_tiles(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self._content)

        columns = 2
        rows = (len(self._tiles) + columns - 1) // columns if self._tiles else 0
        for idx, tile in enumerate(self._tiles):
            row = idx // columns
            col = idx % columns
            self._grid.addWidget(tile.root, row, col)
            # Keep all tiles evenly sized as channel count increases.
            self._grid.setRowStretch(row, 1)
            self._grid.setColumnStretch(col, 1)

        # Clear stale stretch for removed rows/columns.
        for row in range(rows, max(rows, 8)):
            self._grid.setRowStretch(row, 0)
        for col in range(columns, max(columns, 4)):
            self._grid.setColumnStretch(col, 0)
