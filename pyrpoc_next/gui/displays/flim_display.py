"""FLIM lifetime display: global decay curve + threaded per-pixel exponential fit.

Ported from the original pyrpoc FLIM display. Consumes a HistogramCubeParcel (a
(H, W, n_bins) per-pixel decay-histogram cube). The global decay is shown at once,
rolled so the peak lands at t=0; "Render FLIM Image" runs a per-pixel single-exponential
fit (box-summed neighbourhood) on a background thread, filling the lifetime map row by
row. Only the data boundary changed from AcquiredData -> parcel; persistence is deferred.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from scipy.optimize import curve_fit

from pyrpoc_next.gui.displays.base import DisplayWidget, display_registry
from pyrpoc_next.structs.keys import DisplayKey
from pyrpoc_next.structs.manifest import DisplayManifest
from pyrpoc_next.structs.parcels import HistogramCubeParcel, Parcel

pg.setConfigOptions(imageAxisOrder="row-major")

default_binwidth_ps = 100  # 80 MHz / 125 bins fallback


def mono_exp(t: np.ndarray, amplitude: float, tau: float, offset: float) -> np.ndarray:
    return amplitude * np.exp(-t / tau) + offset


def collect_box_histogram(cube: np.ndarray, iy: int, ix: int, half: int) -> np.ndarray:
    """Sum the per-pixel decay histograms over a (2*half+1) box around (iy, ix)."""
    height, width = cube.shape[:2]
    iy0, iy1 = max(0, iy - half), min(height, iy + half + 1)
    ix0, ix1 = max(0, ix - half), min(width, ix + half + 1)
    return cube[iy0:iy1, ix0:ix1, :].sum(axis=(0, 1))


def roll_and_fit(counts: np.ndarray, bin_width_ps: float = 100.0) -> float:
    """Roll a folded decay so the peak is at t=0, fit a single exponential.

    Returns tau in ps, or 0.0 on failure / too few counts.
    """
    if counts.sum() < 5:
        return 0.0
    i_peak = int(np.argmax(counts))
    rolled = np.roll(counts, -i_peak).astype(float)
    t = np.arange(len(rolled)) * bin_width_ps
    amplitude0 = float(rolled[0])
    offset0 = float(np.percentile(rolled, 10))
    tau0 = (t[-1] - t[0]) / 3.0
    sigma = np.maximum(np.sqrt(rolled), 1.0)
    try:
        popt, _ = curve_fit(
            mono_exp, t, rolled, p0=[amplitude0, max(tau0, 1.0), offset0],
            sigma=sigma, absolute_sigma=True,
            bounds=([0.0, 1.0, 0.0], [np.inf, np.inf, np.inf]), maxfev=2000,
        )
        return float(popt[1])
    except Exception:
        return 0.0


class FitWorker(QObject):
    """Runs the per-pixel fitting on a QThread, emitting the partial map per row."""

    row_done = pyqtSignal(object)  # np.ndarray — partial lifetime map
    finished = pyqtSignal()

    def __init__(self, cube: np.ndarray, half: int, binwidth_ps: float):
        super().__init__()
        self.cube = cube
        self.half = half
        self.binwidth_ps = binwidth_ps
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True

    def run(self) -> None:
        cube = self.cube
        height, width = cube.shape[:2]
        lifetime_map = np.zeros((height, width), dtype=np.float32)
        for iy in range(height):
            if self.aborted:
                break
            for ix in range(width):
                counts = collect_box_histogram(cube, iy, ix, self.half)
                if counts.sum() == 0:
                    continue
                lifetime_map[iy, ix] = roll_and_fit(counts, self.binwidth_ps)
            self.row_done.emit(lifetime_map.copy())
        self.finished.emit()


@display_registry.register
class FlimDisplay(DisplayWidget):
    """FLIM lifetime imaging from per-pixel decay histograms."""

    manifest = DisplayManifest(
        key=DisplayKey.flim, display_name="FLIM Lifetime Display",
        accepted_parcels=(HistogramCubeParcel,),
    )

    def __init__(self):
        super().__init__()
        self.raw_frame_hw: np.ndarray | None = None
        self.binwidth_ps: int = default_binwidth_ps
        self.lifetime_hw: np.ndarray | None = None
        self.suspend_lut_signal = False
        self.min_val: float = 0.0
        self.max_val: float = 1.0
        self.fit_thread: QThread | None = None
        self.fit_worker: FitWorker | None = None

        self.lut = pg.ColorMap(
            pos=np.array([0.0, 0.5, 1.0], dtype=float),
            color=np.array([[0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]], dtype=np.ubyte),
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Global decay histogram
        self.trace_plot = pg.PlotWidget()
        self.trace_plot.setMenuEnabled(False)
        self.trace_plot.hideButtons()
        self.trace_plot.setLabel("bottom", "Time since peak", units="ps")
        self.trace_plot.setLabel("left", "Photons")
        self.trace_plot.setFixedHeight(160)
        self.time_trace_curve = self.trace_plot.plot(pen=pg.mkPen(color=(100, 200, 255), width=1))
        outer.addWidget(self.trace_plot)

        # Lifetime image + LUT histogram
        image_row = QHBoxLayout()
        outer.addLayout(image_row, 1)
        image_plot = pg.PlotWidget()
        image_plot.setMenuEnabled(False)
        image_plot.hideButtons()
        image_plot.setAspectLocked(True)
        image_plot.invertY(True)
        self.image_item = pg.ImageItem()
        self.image_item.setColorMap(self.lut)
        image_plot.addItem(self.image_item)
        image_row.addWidget(image_plot, 1)

        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(4)
        self.hist_widget = pg.HistogramLUTWidget()
        self.hist_widget.setImageItem(self.image_item)
        self.hist_widget.item.gradient.setColorMap(self.lut)
        right_col.addWidget(self.hist_widget, 1)
        self.autoscale_box = QCheckBox("Autoscale")
        self.autoscale_box.setChecked(True)
        self.autoscale_box.toggled.connect(self.on_autoscale_toggled)
        right_col.addWidget(self.autoscale_box)
        image_row.addLayout(right_col)
        self.hist_widget.item.sigLevelsChanged.connect(self.on_lut_levels_changed)

        # Fit parameters row
        params_row = QHBoxLayout()
        params_row.setSpacing(8)
        outer.addLayout(params_row)
        params_row.addWidget(QLabel("Box N:"))
        self.box_spin = QSpinBox()
        self.box_spin.setMinimum(1)
        self.box_spin.setMaximum(51)
        self.box_spin.setSingleStep(2)
        self.box_spin.setValue(1)
        self.box_spin.setToolTip("Sum photons from an NxN neighbourhood before fitting each pixel")
        params_row.addWidget(self.box_spin)
        params_row.addStretch(1)
        self.render_button = QPushButton("Render FLIM Image")
        self.render_button.setEnabled(False)
        self.render_button.clicked.connect(self.on_render_clicked)
        params_row.addWidget(self.render_button)

    def handle(self, parcel: Parcel) -> None:
        self.handle_raw_frame(np.asarray(parcel.data), int(parcel.bin_width_ps))

    def clear(self) -> None:
        self.cancel_fit()
        self.raw_frame_hw = None
        self.lifetime_hw = None
        self.time_trace_curve.setData(x=[], y=[])
        self.image_item.setImage(np.zeros((1, 1), dtype=np.float32), autoLevels=False)
        self.render_button.setEnabled(False)
        self.render_button.setText("Render FLIM Image")

    def get_normalized_data_3d(self) -> np.ndarray | None:
        """Float32 [1, H, W] normalized lifetime map — what the mask editor consumes."""
        if self.lifetime_hw is None:
            return None
        arr = self.lifetime_hw
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        norm = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr, dtype=np.float32)
        return np.clip(norm, 0.0, 1.0)[np.newaxis]

    def handle_raw_frame(self, cube: np.ndarray, binwidth_ps: int) -> None:
        self.cancel_fit()
        self.raw_frame_hw = cube
        self.binwidth_ps = binwidth_ps
        counts = np.asarray(cube, dtype=np.float64).sum(axis=(0, 1))
        if counts.sum() > 0:
            i_peak = int(np.argmax(counts))
            rolled = np.roll(counts, -i_peak)
            x = np.arange(len(rolled), dtype=float) * float(binwidth_ps)
            self.time_trace_curve.setData(x=x, y=rolled)
        self.render_button.setEnabled(True)
        self.render_button.setText("Render FLIM Image")

    def on_render_clicked(self) -> None:
        if self.raw_frame_hw is None:
            return
        if self.fit_thread is not None and self.fit_thread.isRunning():
            self.cancel_fit()  # button doubles as cancel while running
            return

        worker = FitWorker(self.raw_frame_hw, self.box_spin.value() // 2, self.binwidth_ps)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.row_done.connect(self.on_row_done)
        worker.finished.connect(self.on_fit_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        self.fit_worker = worker
        self.fit_thread = thread

        height, width = self.raw_frame_hw.shape[:2]
        self.image_item.setImage(np.zeros((height, width), dtype=np.float32), autoLevels=False)
        self.render_button.setText("Cancel")
        thread.start()

    def cancel_fit(self) -> None:
        if self.fit_worker is not None:
            self.fit_worker.abort()
        if self.fit_thread is not None and self.fit_thread.isRunning():
            self.fit_thread.quit()
            self.fit_thread.wait()
        self.fit_worker = None
        self.fit_thread = None

    def on_row_done(self, partial_map: np.ndarray) -> None:
        self.image_item.setImage(partial_map, autoLevels=False)
        if self.autoscale_box.isChecked():
            valid = partial_map[partial_map > 0]
            if valid.size > 0:
                lo, hi = float(valid.min()), float(valid.max())
                if hi <= lo:
                    hi = lo + 1e-12
                self.apply_levels(lo, hi)

    def on_fit_finished(self) -> None:
        final = self.image_item.image
        if final is not None:
            self.lifetime_hw = final.copy()
        self.render_button.setText("Render FLIM Image")
        self.render_button.setEnabled(True)
        self.fit_worker = None
        self.fit_thread = None

    def on_autoscale_toggled(self, checked: bool) -> None:
        if checked and self.lifetime_hw is not None:
            valid = self.lifetime_hw[self.lifetime_hw > 0]
            if valid.size > 0:
                lo, hi = float(valid.min()), float(valid.max())
                if hi <= lo:
                    hi = lo + 1e-12
                self.apply_levels(lo, hi)
        self.request_persist()

    def on_lut_levels_changed(self) -> None:
        if self.suspend_lut_signal:
            return
        lo, hi = self.hist_widget.item.getLevels()
        if hi <= lo:
            hi = lo + 1e-12
            self.hist_widget.item.setLevels(lo, hi)
        self.min_val, self.max_val = float(lo), float(hi)
        self.request_persist()

    def apply_levels(self, lo: float, hi: float) -> None:
        self.min_val, self.max_val = lo, hi
        self.image_item.setLevels((lo, hi))
        self.suspend_lut_signal = True
        try:
            self.hist_widget.item.setLevels(lo, hi)
        finally:
            self.suspend_lut_signal = False
