from __future__ import annotations

from typing import Any, cast

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.backend_utils.acquired_data import AcquiredData, DataKind
from pyrpoc.rpoc.types import RPOCImageInput
from ..base_display import BaseDisplay
from ..display_registry import display_registry
from .controller import AnalysisController
from .panel import AnalysisPanel
from .tile import ChannelTile, build_channel_tile, make_default_lut
from .toolbar import AnalysisTool, AnalysisToolbar


@display_registry.register("tiled_2d")
class Tiled2DDisplay(BaseDisplay):
    display_key = "tiled_2d"
    display_name = "2D Tiled"
    accepted_kinds = [DataKind.INTENSITY_FRAME, DataKind.PARTIAL_FRAME]
    display_parameters = {}

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._data_chw: np.ndarray | None = None
        self._tiles: list[ChannelTile] = []
        self._pending_channel_state: list[dict[str, Any]] = []
        self._pending_analysis: dict[str, Any] | None = None
        self._suspend_lut_signal = False
        self._lut = make_default_lut()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.toolbar = AnalysisToolbar(self)
        outer.addWidget(self.toolbar)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        self._scroll = QScrollArea(splitter)
        self._scroll.setWidgetResizable(True)
        self._content = QWidget(self._scroll)
        self._grid = QGridLayout(self._content)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(10)
        self._scroll.setWidget(self._content)
        splitter.addWidget(self._scroll)

        self.panel = AnalysisPanel(splitter)
        self.panel.setVisible(False)
        splitter.addWidget(self.panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([700, 220])

        self.controller = AnalysisController(self, self.panel)
        self.toolbar.tool_changed.connect(self.on_tool_changed)
        self.toolbar.clear_requested.connect(self.on_clear_requested)

    def configure(self, params: dict[str, Any]) -> None:
        del params

    def render(self, acquired: AcquiredData) -> None:
        self.set_data(acquired.data)

    def clear(self) -> None:
        self._data_chw = None
        self.sync_channel_tiles(0)

    def set_data(self, data_chw: np.ndarray) -> None:
        arr = np.asarray(data_chw, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError("Tiled2DDisplay expects channels-first data [C, H, W]")
        if arr.shape[0] <= 0:
            raise ValueError("Tiled2DDisplay requires at least one channel")
        if arr.shape[1] <= 0 or arr.shape[2] <= 0:
            raise ValueError("Tiled2DDisplay received invalid spatial dimensions")

        self._data_chw = arr
        self.sync_channel_tiles(arr.shape[0])
        for idx in range(arr.shape[0]):
            self.update_channel_image(idx, arr[idx], from_data=True)
        self.controller.refresh()

    # ------------------------------------------------------------------
    # Accessors used by the analysis controller
    # ------------------------------------------------------------------

    def current_data(self) -> np.ndarray | None:
        return self._data_chw

    def tile_at(self, index: int) -> ChannelTile | None:
        if 0 <= index < len(self._tiles):
            return self._tiles[index]
        return None

    def tile_count(self) -> int:
        return len(self._tiles)

    # ------------------------------------------------------------------
    # Analysis toolbar handlers
    # ------------------------------------------------------------------

    def on_tool_changed(self, tool: AnalysisTool) -> None:
        self.controller.set_tool(tool)
        self.panel.setVisible(tool != AnalysisTool.PAN)
        self.request_persist()

    def on_clear_requested(self) -> None:
        self.controller.clear()
        self.toolbar.set_active_tool(AnalysisTool.PAN)
        self.panel.setVisible(False)
        self.request_persist()

    # ------------------------------------------------------------------
    # Channel names / export
    # ------------------------------------------------------------------

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
            source_id=self.display_key,
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

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
        return {"channels": channels, "analysis": self.controller.export_state()}

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
        analysis = state.get("analysis")
        if isinstance(analysis, dict):
            self._pending_analysis = analysis
        self.apply_pending_channel_state()
        self.apply_pending_analysis()

    # ------------------------------------------------------------------
    # Tile management
    # ------------------------------------------------------------------

    def sync_channel_tiles(self, count: int) -> None:
        while len(self._tiles) > count:
            idx = len(self._tiles) - 1
            self.controller.detach_tile(idx)
            tile = self._tiles.pop()
            tile.root.setParent(None)
            tile.root.deleteLater()

        while len(self._tiles) < count:
            idx = len(self._tiles)
            self._tiles.append(self.build_tile(idx))

        self.controller.tiles_synced()
        self.apply_pending_channel_state()
        self.apply_pending_analysis()
        self.reflow_tiles()

    def build_tile(self, index: int) -> ChannelTile:
        tile = build_channel_tile(self._content, index, self._lut)
        tile.name_edit.textChanged.connect(lambda *_args: self.request_persist())
        tile.autoscale_box.toggled.connect(lambda checked, i=index: self.on_autoscale_toggled(i, checked))
        tile.hist_widget.item.sigLevelsChanged.connect(lambda _item, i=index: self.on_lut_levels_changed(i))
        self.controller.register_tile(index, tile)
        return tile

    def on_autoscale_toggled(self, idx: int, checked: bool) -> None:
        if idx < 0 or idx >= len(self._tiles):
            return
        if checked and self._data_chw is not None and idx < self._data_chw.shape[0]:
            self.update_channel_image(idx, self._data_chw[idx], from_data=True)
        elif self._data_chw is not None and idx < self._data_chw.shape[0]:
            tile = self._tiles[idx]
            tile.image_item.setImage(self._data_chw[idx], autoLevels=False)
        self.request_persist()

    def on_lut_levels_changed(self, idx: int) -> None:
        if self._suspend_lut_signal:
            return
        if idx < 0 or idx >= len(self._tiles):
            return
        tile = self._tiles[idx]
        min_val, max_val = cast(tuple[float, float], tile.hist_widget.item.getLevels())
        if max_val <= min_val:
            max_val = min_val + 1e-12
            tile.hist_widget.item.setLevels(min_val, max_val)
        tile.min_val = min_val
        tile.max_val = max_val
        self.request_persist()

    def update_channel_image(self, idx: int, channel: np.ndarray, from_data: bool) -> None:
        del from_data
        tile = self._tiles[idx]
        tile.image_item.setImage(channel, autoLevels=False)

        if tile.autoscale_box.isChecked():
            min_val = float(np.min(channel))
            max_val = float(np.max(channel))
            if max_val <= min_val:
                max_val = min_val + 1e-12
            self.apply_levels(tile, min_val, max_val)
        else:
            min_val, max_val = cast(tuple[float, float], tile.hist_widget.item.getLevels())
            self.apply_levels(tile, min_val, max_val)

    def apply_levels(self, tile: ChannelTile, min_val: float, max_val: float) -> None:
        tile.min_val = float(min_val)
        tile.max_val = float(max_val)
        tile.image_item.setLevels((tile.min_val, tile.max_val))
        self._suspend_lut_signal = True
        try:
            tile.hist_widget.item.setLevels(tile.min_val, tile.max_val)
        finally:
            self._suspend_lut_signal = False

    def apply_pending_channel_state(self) -> None:
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
            self.apply_levels(tile, min_val, max_val)
        # Apply persisted state once; live user changes during acquisition
        # should not be overwritten each frame.
        self._pending_channel_state = []

    def apply_pending_analysis(self) -> None:
        if self._pending_analysis is None or not self._tiles:
            return
        state = self._pending_analysis
        self._pending_analysis = None
        self.controller.restore(state)
        tool = self.controller.current_tool()
        self.toolbar.set_active_tool(tool)
        self.panel.setVisible(tool != AnalysisTool.PAN)

    def reflow_tiles(self) -> None:
        while (item := self._grid.takeAt(0)) is not None:
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
