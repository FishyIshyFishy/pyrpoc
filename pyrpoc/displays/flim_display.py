from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from scipy.optimize import curve_fit
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.backend_utils.acquired_data import AcquiredData, DataKind
from pyrpoc.rpoc.types import RPOCImageInput
from .base_display import BaseDisplay
from .display_registry import display_registry

_DEFAULT_LASER_PERIOD_PS = 12500  # 80 MHz fallback


def _mono_exp(t: np.ndarray, A: float, tau: float, B: float) -> np.ndarray:
    return A * np.exp(-t / tau) + B


def _collect_box_delays(raw: np.ndarray, iy: int, ix: int, half: int) -> np.ndarray:
    H, W = raw.shape
    iy0, iy1 = max(0, iy - half), min(H, iy + half + 1)
    ix0, ix1 = max(0, ix - half), min(W, ix + half + 1)
    cells = [
        raw[r, c]
        for r in range(iy0, iy1)
        for c in range(ix0, ix1)
        if len(raw[r, c]) > 0
    ]
    return np.concatenate(cells).astype(np.int64) if cells else np.empty(0, dtype=np.int64)


def _roll_and_fit(counts: np.ndarray, bin_width_ps: float = 100.0) -> float:
    """Roll a folded decay histogram so the peak is at t=0, then fit a
    single exponential.  Returns tau in ps, or 0.0 on failure."""
    if counts.sum() < 5:
        return 0.0

    i_peak = int(np.argmax(counts))
    rolled = np.roll(counts, -i_peak).astype(float)
    t = np.arange(len(rolled)) * bin_width_ps

    A0 = float(rolled[0])
    B0 = float(np.percentile(rolled, 10))
    tau0 = (t[-1] - t[0]) / 3.0
    sigma = np.maximum(np.sqrt(rolled), 1.0)

    try:
        popt, _ = curve_fit(
            _mono_exp,
            t,
            rolled,
            p0=[A0, max(tau0, 1.0), B0],
            sigma=sigma,
            absolute_sigma=True,
            bounds=([0.0, 1.0, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=2000,
        )
        return float(popt[1])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _FitWorker(QObject):
    """Runs the per-pixel fitting on a QThread.

    Emits ``row_done`` with a copy of the partial lifetime map after every
    row so the display can update progressively.  Emits ``finished`` when
    all rows are complete or the worker is aborted.
    """

    row_done = pyqtSignal(object)   # np.ndarray — partial lifetime map
    finished = pyqtSignal()

    def __init__(
        self,
        raw: np.ndarray,
        half: int,
        bins: np.ndarray,
    ) -> None:
        super().__init__()
        self._raw = raw
        self._half = half
        self._bins = bins
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        raw = self._raw
        H, W = raw.shape
        half = self._half
        bins = self._bins
        lifetime_map = np.zeros((H, W), dtype=np.float32)

        for iy in range(H):
            if self._abort:
                break
            for ix in range(W):
                delays = _collect_box_delays(raw, iy, ix, half)
                if delays.size == 0:
                    continue
                counts, _ = np.histogram(delays, bins=bins)
                lifetime_map[iy, ix] = _roll_and_fit(counts)

            self.row_done.emit(lifetime_map.copy())

        self.finished.emit()


# ---------------------------------------------------------------------------
# Display widget
# ---------------------------------------------------------------------------

@display_registry.register("flim_display")
class FlimDisplay(BaseDisplay):
    """FLIM display.

    When a FLIM_RAW_FRAME arrives the global decay histogram is shown
    immediately, rolled so the peak lands at t=0.  Click
    "Render FLIM Image" to run a per-pixel single-exponential fit using
    the box-summing parameter.  The lifetime image fills in row by row
    while fitting runs in a background thread.
    """

    DISPLAY_KEY = "flim_display"
    DISPLAY_NAME = "FLIM Lifetime Display"
    ACCEPTED_KINDS = [DataKind.FLIM_RAW_FRAME]
    DISPLAY_PARAMETERS = {}

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        pg.setConfigOptions(imageAxisOrder="row-major")

        self._raw_frame_hw: np.ndarray | None = None
        self._laser_period_ps: int = _DEFAULT_LASER_PERIOD_PS
        self._lifetime_hw: np.ndarray | None = None
        self._suspend_lut_signal = False
        self._min_val: float = 0.0
        self._max_val: float = 1.0
        self._autoscale: bool = True
        self._pending_state: dict[str, Any] = {}

        self._fit_thread: QThread | None = None
        self._fit_worker: _FitWorker | None = None

        self._lut = pg.ColorMap(
            pos=np.array([0.0, 0.5, 1.0], dtype=float),
            color=np.array(
                [[0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]],
                dtype=np.ubyte,
            ),
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # --- Global decay histogram ---
        self._trace_plot = pg.PlotWidget()
        self._trace_plot.setMenuEnabled(False)
        self._trace_plot.hideButtons()
        self._trace_plot.setLabel("bottom", "Time since peak", units="ps")
        self._trace_plot.setLabel("left", "Photons")
        self._trace_plot.setFixedHeight(160)
        self._time_trace_curve = self._trace_plot.plot(
            pen=pg.mkPen(color=(100, 200, 255), width=1)
        )
        outer.addWidget(self._trace_plot)

        # --- Lifetime image + LUT histogram ---
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

        # --- Fit parameters row ---
        params_row = QHBoxLayout()
        params_row.setSpacing(8)
        outer.addLayout(params_row)

        params_row.addWidget(QLabel("Box N:"))
        self._box_spin = QSpinBox()
        self._box_spin.setMinimum(1)
        self._box_spin.setMaximum(51)
        self._box_spin.setSingleStep(2)
        self._box_spin.setValue(1)
        self._box_spin.setToolTip(
            "Sum photons from an NxN neighbourhood before fitting each pixel"
        )
        params_row.addWidget(self._box_spin)
        params_row.addStretch(1)

        self._render_button = QPushButton("Render FLIM Image")
        self._render_button.setEnabled(False)
        self._render_button.clicked.connect(self._on_render_clicked)
        params_row.addWidget(self._render_button)

        if self._pending_state:
            self._apply_pending_state()

    # ------------------------------------------------------------------
    # BaseDisplay interface
    # ------------------------------------------------------------------

    def configure(self, params: dict[str, Any]) -> None:
        del params

    def render(self, acquired: AcquiredData) -> None:
        if acquired.kind == DataKind.FLIM_RAW_FRAME:
            lp = acquired.metadata.get("laser_period_ps", _DEFAULT_LASER_PERIOD_PS)
            self._handle_raw_frame(acquired.data, int(lp))

    def clear(self) -> None:
        self._cancel_fit()
        self._raw_frame_hw = None
        self._lifetime_hw = None
        self._time_trace_curve.setData(x=[], y=[])
        blank = np.zeros((1, 1), dtype=np.float32)
        self._image_item.setImage(blank, autoLevels=False)
        self._render_button.setEnabled(False)
        self._render_button.setText("Render FLIM Image")

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
            "box_n": self._box_spin.value(),
        }

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._pending_state = state
        self._apply_pending_state()

    # ------------------------------------------------------------------
    # Internal — data handler
    # ------------------------------------------------------------------

    def _handle_raw_frame(self, raw: np.ndarray, laser_period_ps: int) -> None:
        self._cancel_fit()
        self._raw_frame_hw = raw
        self._laser_period_ps = laser_period_ps

        n_bins = laser_period_ps // 100
        bins = np.arange(0, n_bins * 100 + 100, 100, dtype=np.int64)

        cells = [cell for cell in raw.flat if len(cell) > 0]
        if cells:
            all_delays = np.concatenate(cells).astype(np.int64)
            counts, _ = np.histogram(all_delays, bins=bins)
            i_peak = int(np.argmax(counts))
            rolled = np.roll(counts, -i_peak)
            x = np.arange(len(rolled), dtype=float) * 100.0
            self._time_trace_curve.setData(x=x, y=rolled.astype(float))

        self._render_button.setEnabled(True)
        self._render_button.setText("Render FLIM Image")

    # ------------------------------------------------------------------
    # Internal — threaded render
    # ------------------------------------------------------------------

    def _on_render_clicked(self) -> None:
        if self._raw_frame_hw is None:
            return

        # If already rendering, treat the button as a cancel
        if self._fit_thread is not None and self._fit_thread.isRunning():
            self._cancel_fit()
            return

        n_bins = self._laser_period_ps // 100
        bins = np.arange(0, n_bins * 100 + 100, 100, dtype=np.int64)

        worker = _FitWorker(
            raw=self._raw_frame_hw,
            half=self._box_spin.value() // 2,
            bins=bins,
        )
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.row_done.connect(self._on_row_done)
        worker.finished.connect(self._on_fit_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)

        self._fit_worker = worker
        self._fit_thread = thread

        # Clear the image so progress is visible from the start
        H, W = self._raw_frame_hw.shape
        blank = np.zeros((H, W), dtype=np.float32)
        self._image_item.setImage(blank, autoLevels=False)

        self._render_button.setText("Cancel")
        thread.start()

    def _cancel_fit(self) -> None:
        if self._fit_worker is not None:
            self._fit_worker.abort()
        if self._fit_thread is not None and self._fit_thread.isRunning():
            self._fit_thread.quit()
            self._fit_thread.wait()
        self._fit_worker = None
        self._fit_thread = None

    def _on_row_done(self, partial_map: np.ndarray) -> None:
        self._image_item.setImage(partial_map, autoLevels=False)
        if self._autoscale_box.isChecked():
            valid = partial_map[partial_map > 0]
            if valid.size > 0:
                lo, hi = float(valid.min()), float(valid.max())
                if hi <= lo:
                    hi = lo + 1e-12
                self._apply_levels(lo, hi)

    def _on_fit_finished(self) -> None:
        final = self._image_item.image
        if final is not None:
            self._lifetime_hw = final.copy()
        self._render_button.setText("Render FLIM Image")
        self._render_button.setEnabled(True)
        self._fit_worker = None
        self._fit_thread = None

    # ------------------------------------------------------------------
    # Internal — LUT helpers
    # ------------------------------------------------------------------

    def _on_autoscale_toggled(self, checked: bool) -> None:
        self._autoscale = checked
        if checked and self._lifetime_hw is not None:
            valid = self._lifetime_hw[self._lifetime_hw > 0]
            if valid.size > 0:
                lo, hi = float(valid.min()), float(valid.max())
                if hi <= lo:
                    hi = lo + 1e-12
                self._apply_levels(lo, hi)
        self.request_persist()

    def _on_lut_levels_changed(self) -> None:
        if self._suspend_lut_signal:
            return
        lo, hi = self._hist_widget.item.getLevels()
        if hi <= lo: #pyright:ignore
            hi = lo + 1e-12 #pyright:ignore
            self._hist_widget.item.setLevels(lo, hi)
        self._min_val = float(lo) #pyright:ignore
        self._max_val = float(hi) #pyright:ignore
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
        if "box_n" in state:
            self._box_spin.setValue(int(state["box_n"]))
        self._pending_state = {}
