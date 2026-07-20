"""FLIM display: the lifetime fit math and the frame-handling path."""

from __future__ import annotations

import numpy as np

from pyrpoc_next.structs.parcels import HistogramCubeParcel


def exp_decay(n_bins: int, tau_ps: float, bin_width_ps: float, amplitude: float = 2000.0) -> np.ndarray:
    t = np.arange(n_bins, dtype=float) * bin_width_ps
    return amplitude * np.exp(-t / tau_ps) + 5.0


def test_roll_and_fit_recovers_tau():
    from pyrpoc_next.gui.displays.flim_display import roll_and_fit

    bw = 100.0
    counts = exp_decay(125, tau_ps=2000.0, bin_width_ps=bw)
    tau = roll_and_fit(counts, bw)
    assert abs(tau - 2000.0) / 2000.0 < 0.1  # within 10% of the true lifetime


def test_roll_and_fit_returns_zero_on_empty():
    from pyrpoc_next.gui.displays.flim_display import roll_and_fit

    assert roll_and_fit(np.zeros(64), 100.0) == 0.0


def test_handle_frame_enables_render_and_plots_decay(qapp):
    from pyrpoc_next.gui.displays import FlimDisplay

    display = FlimDisplay()
    cube = np.tile(exp_decay(64, 1500.0, 100.0), (6, 6, 1)).astype(np.float32)
    display.handle(HistogramCubeParcel(data=cube, bin_width_ps=100.0, laser_period_ps=12500.0))

    assert display.render_button.isEnabled()
    x, y = display.time_trace_curve.getData()
    assert x is not None and len(x) == 64  # global decay trace populated


def test_fit_worker_produces_lifetime_map(qapp):
    from pyrpoc_next.gui.displays.flim_display import FitWorker

    cube = np.tile(exp_decay(64, 1800.0, 100.0), (4, 4, 1)).astype(np.float32)
    rows = []
    worker = FitWorker(cube, half=0, binwidth_ps=100.0)
    worker.row_done.connect(lambda m: rows.append(m))
    worker.run()  # synchronous (not moved to a thread)

    assert len(rows) == 4  # one emit per row
    final = rows[-1]
    assert final.shape == (4, 4)
    assert np.all(final > 0)  # every pixel fit to a positive lifetime
    assert abs(float(final.mean()) - 1800.0) / 1800.0 < 0.15
