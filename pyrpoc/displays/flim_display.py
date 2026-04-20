from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.backend_utils.acquired_data import AcquiredData, DataKind
from pyrpoc.rpoc.types import RPOCImageInput
from .base_display import BaseDisplay
from .display_registry import display_registry


@display_registry.register("flim_display")
class FlimDisplay(BaseDisplay):
    """FLIM-specific display with a live time-trace and a post-acquisition
    lifetime image renderer.

    During acquisition, FLIM_PARTIAL_HISTOGRAM emissions update the decay
    histogram plot at ~10 Hz. After acquisition completes, a FLIM_RAW_FRAME
    emission delivers per-pixel delay arrays; the "Render FLIM Image" button
    then computes the mean-lifetime image and shows it.
    """

    DISPLAY_KEY = "flim_display"
    DISPLAY_NAME = "FLIM Lifetime Display"
    ACCEPTED_KINDS = [DataKind.FLIM_RAW_FRAME, DataKind.FLIM_PARTIAL_HISTOGRAM]
    DISPLAY_PARAMETERS = {}

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        pg.setConfigOptions(imageAxisOrder="row-major")

        self._raw_frame_hw: np.ndarray | None = None
        self._lifetime_hw: np.ndarray | None = None
        self._suspend_lut_signal = False
        self._min_val: float = 0.0
        self._max_val: float = 1.0
        self._autoscale: bool = True
        self._pending_state: dict[str, Any] = {}

        self._lut = pg.ColorMap(
            pos=np.array([0.0, 0.5, 1.0], dtype=float),
            color=np.array(
                [
                    [0, 0, 255, 255],
                    [0, 255, 0, 255],
                    [255, 0, 0, 255],
                ],
                dtype=np.ubyte,
            ),
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # --- Time-trace plot ---
        self._trace_plot = pg.PlotWidget()
        self._trace_plot.setMenuEnabled(False)
        self._trace_plot.hideButtons()
        self._trace_plot.setLabel("bottom", "Delay", units="ps")
        self._trace_plot.setLabel("left", "Photons")
        self._trace_plot.setFixedHeight(160)
        self._time_trace_curve = self._trace_plot.plot(
            pen=pg.mkPen(color=(100, 200, 255), width=1)
        )
        outer.addWidget(self._trace_plot)

        # --- Lifetime image + histogram ---
        image_row = QHBoxLayout()
        outer.addLayout(image_row, 1)

        image_plot = pg.PlotWidget()
        image_plot.setMenuEnabled(False)
        image_plot.hideButtons()
        image_plot.setAspectLocked(True)
        image_plot.invertY(True)
        self._image_item = pg.ImageItem()
        self._image_item.setColorMap(self._lut)
        image_plot.addItem(self._image_item)
        image_row.addWidget(image_plot, 1)

        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(4)

        self._hist_widget = pg.HistogramLUTWidget()
        self._hist_widget.setImageItem(self._image_item)
        self._hist_widget.item.gradient.setColorMap(self._lut)
        right_col.addWidget(self._hist_widget, 1)

        self._autoscale_box = QCheckBox("Autoscale")
        self._autoscale_box.setChecked(True)
        self._autoscale_box.toggled.connect(self._on_autoscale_toggled)
        right_col.addWidget(self._autoscale_box)

        image_row.addLayout(right_col)

        self._hist_widget.item.sigLevelsChanged.connect(self._on_lut_levels_changed)

        # --- Render button ---
        self._render_button = QPushButton("Render FLIM Image")
        self._render_button.setEnabled(False)
        self._render_button.clicked.connect(self._on_render_clicked)
        outer.addWidget(self._render_button)

        if self._pending_state:
            self._apply_pending_state()

    # ------------------------------------------------------------------
    # BaseDisplay interface
    # ------------------------------------------------------------------

    def configure(self, params: dict[str, Any]) -> None:
        del params

    def render(self, acquired: AcquiredData) -> None:
        if acquired.kind == DataKind.FLIM_PARTIAL_HISTOGRAM:
            self._handle_partial_histogram(acquired.data)
        elif acquired.kind == DataKind.FLIM_RAW_FRAME:
            self._handle_raw_frame(acquired.data)

    def clear(self) -> None:
        self._raw_frame_hw = None
        self._lifetime_hw = None
        self._time_trace_curve.setData(x=[], y=[])
        blank = np.zeros((1, 1), dtype=np.float32)
        self._image_item.setImage(blank, autoLevels=False)
        self._render_button.setEnabled(False)

    # ------------------------------------------------------------------
    # RPOC / normalized export
    # ------------------------------------------------------------------

    def export_rpoc_input(self) -> RPOCImageInput | None:
        if self._lifetime_hw is None:
            return None
        chw = self._lifetime_hw[np.newaxis].astype(np.float32, copy=True)
        return RPOCImageInput(
            data=chw,
            channel_labels=["lifetime"],
            source_id=self.DISPLAY_KEY,
        )

    def get_normalized_data_3d(self) -> np.ndarray | None:
        if self._lifetime_hw is None:
            return None
        arr = self._lifetime_hw
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi > lo:
            norm = (arr - lo) / (hi - lo)
        else:
            norm = np.zeros_like(arr, dtype=np.float32)
        return np.clip(norm, 0.0, 1.0)[np.newaxis]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def export_persistence_state(self) -> dict[str, Any]:
        return {
            "autoscale": self._autoscale_box.isChecked(),
            "min_val": self._min_val,
            "max_val": self._max_val,
        }

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._pending_state = state
        self._apply_pending_state()

    # ------------------------------------------------------------------
    # Internal — data handlers
    # ------------------------------------------------------------------

    def _handle_partial_histogram(self, hist: np.ndarray) -> None:
        x = np.arange(len(hist), dtype=float) * 100.0
        self._time_trace_curve.setData(x=x, y=hist.astype(float))

    def _handle_raw_frame(self, raw: np.ndarray) -> None:
        self._raw_frame_hw = raw
        # Build final global histogram from all pixel delay lists
        cells = [cell for cell in raw.flat if len(cell) > 0]
        if cells:
            all_delays = np.concatenate(cells).astype(np.int64)
            if all_delays.size > 0:
                bin_max = int(all_delays.max()) + 100
                bins = np.arange(0, bin_max, 100, dtype=np.int64)
                counts, _ = np.histogram(all_delays, bins=bins)
                x = np.arange(len(counts), dtype=float) * 100.0
                self._time_trace_curve.setData(x=x, y=counts.astype(float))
        self._render_button.setEnabled(True)

    def _on_render_clicked(self) -> None:
        if self._raw_frame_hw is None:
            return
        H, W = self._raw_frame_hw.shape
        lifetime_map = np.zeros((H, W), dtype=np.float32)
        for iy in range(H):
            for ix in range(W):
                delays = self._raw_frame_hw[iy, ix]
                if len(delays) > 0:
                    lifetime_map[iy, ix] = float(np.mean(delays))
        self._lifetime_hw = lifetime_map
        self._image_item.setImage(lifetime_map, autoLevels=False)
        if self._autoscale_box.isChecked():
            lo = float(np.nanmin(lifetime_map))
            hi = float(np.nanmax(lifetime_map))
            if hi <= lo:
                hi = lo + 1e-12
            self._apply_levels(lo, hi)
        else:
            lo, hi = self._hist_widget.item.getLevels()
            self._apply_levels(float(lo), float(hi))

    # ------------------------------------------------------------------
    # Internal — LUT helpers
    # ------------------------------------------------------------------

    def _on_autoscale_toggled(self, checked: bool) -> None:
        self._autoscale = checked
        if checked and self._lifetime_hw is not None:
            lo = float(np.nanmin(self._lifetime_hw))
            hi = float(np.nanmax(self._lifetime_hw))
            if hi <= lo:
                hi = lo + 1e-12
            self._apply_levels(lo, hi)
        self.request_persist()

    def _on_lut_levels_changed(self) -> None:
        if self._suspend_lut_signal:
            return
        lo, hi = self._hist_widget.item.getLevels()
        if hi <= lo:
            hi = lo + 1e-12
            self._hist_widget.item.setLevels(lo, hi)
        self._min_val = float(lo)
        self._max_val = float(hi)
        self.request_persist()

    def _apply_levels(self, lo: float, hi: float) -> None:
        self._min_val = lo
        self._max_val = hi
        self._image_item.setLevels((lo, hi))
        self._suspend_lut_signal = True
        try:
            self._hist_widget.item.setLevels(lo, hi)
        finally:
            self._suspend_lut_signal = False

    def _apply_pending_state(self) -> None:
        state = self._pending_state
        if not state:
            return
        autoscale = bool(state.get("autoscale", True))
        lo = float(state.get("min_val", self._min_val))
        hi = float(state.get("max_val", self._max_val))
        self._autoscale_box.blockSignals(True)
        self._autoscale_box.setChecked(autoscale)
        self._autoscale_box.blockSignals(False)
        self._autoscale = autoscale
        if hi <= lo:
            hi = lo + 1e-12
        self._apply_levels(lo, hi)
        self._pending_state = {}
