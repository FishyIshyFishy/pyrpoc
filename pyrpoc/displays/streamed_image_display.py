from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.backend_utils.acquired_data import AcquiredData, DataKind
from pyrpoc.rpoc.types import RPOCImageInput
from .base_display import BaseDisplay
from .display_registry import display_registry


@dataclass
class _ImageTile:
    root: QWidget
    name_edit: QLineEdit
    autoscale_box: QCheckBox
    image_item: pg.ImageItem
    hist_widget: pg.HistogramLUTWidget
    min_val: float = 0.0
    max_val: float = 1.0


@display_registry.register("streamed_image")
class StreamedImageDisplay(BaseDisplay):
    """Live-updating progressive image display.

    Accepts PARTIAL_FRAME (progressive pixel updates during a scan) and
    INTENSITY_FRAME (final complete frame) from any modality. Both carry
    ``(C, H, W)`` or ``(1, H, W)`` float32 data; only the first channel is
    displayed.

    Blank-region mitigation: rows that have zero photon counts in a partial
    frame are filled from the previous complete INTENSITY_FRAME so the
    unscanned portion shows historical data rather than black.
    """

    DISPLAY_KEY = "streamed_image"
    DISPLAY_NAME = "Single Channel Streamed"
    ACCEPTED_KINDS = [DataKind.INTENSITY_FRAME, DataKind.PARTIAL_FRAME]
    DISPLAY_PARAMETERS = {}

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        pg.setConfigOptions(imageAxisOrder="row-major")

        self._data_hw: np.ndarray | None = None
        self._last_complete_hw: np.ndarray | None = None
        self._tile: _ImageTile | None = None
        self._suspend_lut_signal = False
        self._pending_channel_state: list[dict[str, Any]] = []

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
        outer.setContentsMargins(8, 8, 8, 8)
        self._tile = self._build_tile()
        outer.addWidget(self._tile.root)

    # ------------------------------------------------------------------
    # BaseDisplay interface
    # ------------------------------------------------------------------

    def configure(self, params: dict[str, Any]) -> None:
        del params

    def render(self, acquired: AcquiredData) -> None:
        data = acquired.data
        if data.ndim == 3 and data.shape[0] >= 1:
            frame = data[0]
        elif data.ndim == 2:
            frame = data
        else:
            return

        frame = np.asarray(frame, dtype=np.float32)

        if acquired.kind == DataKind.PARTIAL_FRAME:
            self._update_partial(frame)
        else:
            self._update_final(frame)

    def clear(self) -> None:
        self._data_hw = None
        self._last_complete_hw = None
        if self._tile is not None:
            blank = np.zeros((1, 1), dtype=np.float32)
            self._tile.image_item.setImage(blank, autoLevels=False)

    # ------------------------------------------------------------------
    # RPOC / normalized export
    # ------------------------------------------------------------------

    def export_rpoc_input(self) -> RPOCImageInput | None:
        if self._data_hw is None:
            return None
        chw = self._data_hw[np.newaxis].astype(np.float32, copy=True)
        return RPOCImageInput(
            data=chw,
            channel_labels=self.get_channel_names(),
            source_id=self.DISPLAY_KEY,
        )

    def get_normalized_data_3d(self) -> np.ndarray | None:
        if self._data_hw is None:
            return None
        arr = self._data_hw
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi > lo:
            norm = (arr - lo) / (hi - lo)
        else:
            norm = np.zeros_like(arr, dtype=np.float32)
        return np.clip(norm, 0.0, 1.0)[np.newaxis]

    def get_channel_names(self) -> list[str]:
        if self._tile is None:
            return ["Intensity"]
        return [self._tile.name_edit.text().strip() or "Intensity"]

    def set_channel_names(self, names: list[str]) -> None:
        if self._tile is not None and names:
            self._tile.name_edit.setText(str(names[0]).strip())
        self.request_persist()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def export_persistence_state(self) -> dict[str, Any]:
        if self._tile is None:
            return {}
        return {
            "name": self._tile.name_edit.text().strip(),
            "autoscale": bool(self._tile.autoscale_box.isChecked()),
            "min_val": float(self._tile.min_val),
            "max_val": float(self._tile.max_val),
        }

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._pending_channel_state = [state]
        self._apply_pending_state()

    # ------------------------------------------------------------------
    # Internal — frame update paths
    # ------------------------------------------------------------------

    def _update_partial(self, frame: np.ndarray) -> None:
        """Paint an in-progress frame; fill unscanned (zero) rows from the
        previous complete frame to avoid a black lower half during scanning."""
        if self._last_complete_hw is not None and self._last_complete_hw.shape == frame.shape:
            blended = frame.copy()
            empty_rows = frame.sum(axis=1) == 0
            blended[empty_rows] = self._last_complete_hw[empty_rows]
        else:
            blended = frame

        self._data_hw = blended
        tile = self._tile
        if tile is None:
            return
        tile.image_item.setImage(blended, autoLevels=False)
        if tile.autoscale_box.isChecked():
            hi = float(np.max(blended))
            if hi > tile.max_val:
                self._apply_levels(tile, 0.0, hi)

    def _update_final(self, frame: np.ndarray) -> None:
        """Paint the finished frame, update the reference for blank-region fill,
        and (if autoscale) fit levels to the full range."""
        self._data_hw = frame
        self._last_complete_hw = frame
        tile = self._tile
        if tile is None:
            return
        tile.image_item.setImage(frame, autoLevels=False)
        if tile.autoscale_box.isChecked():
            lo = float(np.min(frame))
            hi = float(np.max(frame))
            if hi <= lo:
                hi = lo + 1e-12
            self._apply_levels(tile, lo, hi)
        else:
            lo, hi = tile.hist_widget.item.getLevels()
            self._apply_levels(tile, float(lo), float(hi))

    # ------------------------------------------------------------------
    # Internal — widget helpers
    # ------------------------------------------------------------------

    def _build_tile(self) -> _ImageTile:
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

        tile = _ImageTile(
            root=root,
            name_edit=name_edit,
            autoscale_box=autoscale_box,
            image_item=image_item,
            hist_widget=hist_widget,
        )
        autoscale_box.toggled.connect(self._on_autoscale_toggled)
        hist_widget.item.sigLevelsChanged.connect(self._on_lut_levels_changed)
        return tile

    def _on_autoscale_toggled(self, checked: bool) -> None:
        tile = self._tile
        if tile is None or self._data_hw is None:
            return
        if checked:
            lo = float(np.min(self._data_hw))
            hi = float(np.max(self._data_hw))
            if hi <= lo:
                hi = lo + 1e-12
            self._apply_levels(tile, lo, hi)
        self.request_persist()

    def _on_lut_levels_changed(self) -> None:
        if self._suspend_lut_signal:
            return
        tile = self._tile
        if tile is None:
            return
        lo, hi = tile.hist_widget.item.getLevels()
        if hi <= lo:
            hi = lo + 1e-12
            tile.hist_widget.item.setLevels(lo, hi)
        tile.min_val = float(lo)
        tile.max_val = float(hi)
        self.request_persist()

    def _apply_levels(self, tile: _ImageTile, lo: float, hi: float) -> None:
        tile.min_val = lo
        tile.max_val = hi
        tile.image_item.setLevels((lo, hi))
        self._suspend_lut_signal = True
        try:
            tile.hist_widget.item.setLevels(lo, hi)
        finally:
            self._suspend_lut_signal = False

    def _apply_pending_state(self) -> None:
        if not self._pending_channel_state or self._tile is None:
            return
        state = self._pending_channel_state[0]
        tile = self._tile
        name = str(state.get("name", "")).strip()
        autoscale = bool(state.get("autoscale", True))
        lo = float(state.get("min_val", tile.min_val))
        hi = float(state.get("max_val", tile.max_val))
        tile.name_edit.blockSignals(True)
        tile.name_edit.setText(name or "Intensity")
        tile.name_edit.blockSignals(False)
        tile.autoscale_box.blockSignals(True)
        tile.autoscale_box.setChecked(autoscale)
        tile.autoscale_box.blockSignals(False)
        if hi <= lo:
            hi = lo + 1e-12
        self._apply_levels(tile, lo, hi)
        self._pending_channel_state = []


@display_registry.register("flim_2d")
class Flim2DDisplay(StreamedImageDisplay):
    """Backward-compatibility alias for sessions saved with the 'flim_2d' key."""

    DISPLAY_KEY = "flim_2d"
    DISPLAY_NAME = "FLIM 2D Display"
