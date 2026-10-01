"""Display levels for pyqtgraph histograms."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg


def mono_levels(hist_widget: pg.HistogramLUTWidget) -> tuple[float, float]:
    """The (min, max) levels of a mono-mode histogram. pyqtgraph's
    ``getLevels`` is untyped and also covers rgba mode, so it is read through
    numpy rather than trusted as a pair of floats."""
    lo, hi = np.asarray(hist_widget.item.getLevels(), dtype=np.float64).ravel()[:2]
    return float(lo), float(hi)


def autoscale_levels(channel: np.ndarray) -> tuple[float, float]:
    """Autoscale bounds for one channel, never degenerate."""
    lo = float(np.min(channel))
    hi = float(np.max(channel))
    return lo, max(hi, lo + 1e-12)
