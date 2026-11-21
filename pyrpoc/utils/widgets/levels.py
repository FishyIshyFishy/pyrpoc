"""Display levels for pyqtgraph images: the histogram control that sets them, and the bounds
autoscale picks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

# The smallest gap kept between the low and high level, so scaling never divides by zero.
MIN_SPAN = 1e-12


def mono_levels(hist_widget: pg.HistogramLUTWidget) -> tuple[float, float]:
    """The (min, max) levels of a mono-mode histogram."""
    # pyqtgraph's ``getLevels`` is untyped and also covers rgba mode, so it is read through numpy
    # rather than trusted as a pair of floats.
    lo, hi = np.asarray(hist_widget.item.getLevels(), dtype=np.float64).ravel()[:2]
    return float(lo), float(hi)


def autoscale_levels(channel: np.ndarray) -> tuple[float, float]:
    """Autoscale bounds for one channel, never degenerate."""
    lo = float(np.min(channel))
    hi = float(np.max(channel))
    return lo, max(hi, lo + MIN_SPAN)


@dataclass
class LevelsState:
    """One channel's saved levels and autoscale."""

    autoscale: bool
    min_val: float
    max_val: float

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> LevelsState:
        return cls(
            autoscale=bool(row.get("autoscale", True)),
            min_val=float(row.get("min_val", 0.0)),
            max_val=float(row.get("max_val", 1.0)),
        )

    def to_json(self) -> dict[str, Any]:
        return {"autoscale": self.autoscale, "min_val": self.min_val, "max_val": self.max_val}


def parse_levels(raw: Any) -> list[LevelsState]:
    """Saved levels from session JSON, one per channel in order, skipping malformed rows."""
    if not isinstance(raw, list):
        return []
    return [LevelsState.from_json(row) for row in raw if isinstance(row, dict)]


class LevelsControl(QWidget):
    """An autoscale box above a histogram whose handles set the levels ``image_item`` is drawn
    with, through ``color_map``. Panels may put a channel's name in ``header``, above both."""

    # The user dragged the levels, toggled autoscale or picked another colour map.
    changed = pyqtSignal()

    def __init__(self, parent: QWidget, image_item: pg.ImageItem, color_map: pg.ColorMap):
        super().__init__(parent)
        self.image_item = image_item
        # The bounds autoscale goes to, given with the latest plane.
        self.auto: tuple[float, float] | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.header = QHBoxLayout()
        layout.addLayout(self.header)
        self.autoscale = QCheckBox("Autoscale", self)
        self.autoscale.setChecked(True)
        self.autoscale.setToolTip("Rescale the levels to each new image")
        layout.addWidget(self.autoscale)
        image_item.setColorMap(color_map)
        self.histogram = pg.HistogramLUTWidget(self)
        self.histogram.setImageItem(image_item)
        self.histogram.item.gradient.setColorMap(color_map)
        layout.addWidget(self.histogram, 1)
        self.autoscale.toggled.connect(self.on_autoscale_toggled)
        self.histogram.item.sigLevelsChanged.connect(lambda _item: self.on_levels_dragged())
        self.histogram.item.sigLookupTableChanged.connect(lambda _item: self.changed.emit())

    def take(self, plane: np.ndarray, auto: tuple[float, float]) -> None:
        """Show ``plane``, rescaled to ``auto`` while autoscale is on and otherwise kept at the
        levels set."""
        self.auto = auto
        low, high = auto if self.autoscale.isChecked() else self.levels()
        self.image_item.setImage(plane, autoLevels=False)
        self.set_levels(low, high)

    def levels(self) -> tuple[float, float]:
        return mono_levels(self.histogram)

    def set_levels(self, low: float, high: float) -> None:
        high = max(high, low + MIN_SPAN)
        # Blocked so levels set here are not mistaken for the user dragging them; the image
        # still follows, as it listens to the histogram's region rather than this signal.
        self.histogram.item.blockSignals(True)
        self.histogram.item.setLevels(low, high)
        self.histogram.item.blockSignals(False)
        self.image_item.setLevels((low, high))

    def lookup_table(self) -> np.ndarray:
        """The colour map as a ``(256, 3)`` float table in 0..1."""
        table = self.histogram.item.gradient.colorMap().getLookupTable(0.0, 1.0, 256, alpha=False)
        return np.asarray(table, dtype=np.float32) / 255.0

    def scaled(self, plane: np.ndarray) -> np.ndarray:
        """``plane`` mapped from the levels to 0..1, clipped."""
        low, high = self.levels()
        return np.clip((plane - low) / max(high - low, MIN_SPAN), 0.0, 1.0)

    def on_levels_dragged(self) -> None:
        low, high = self.levels()
        if high <= low:
            self.set_levels(low, high)
        self.changed.emit()

    def on_autoscale_toggled(self, checked: bool) -> None:
        if checked and self.auto is not None:
            self.set_levels(*self.auto)
        self.changed.emit()

    def export_state(self) -> LevelsState:
        low, high = self.levels()
        return LevelsState(self.autoscale.isChecked(), low, high)

    def restore(self, saved: LevelsState) -> None:
        self.autoscale.blockSignals(True)
        self.autoscale.setChecked(saved.autoscale)
        self.autoscale.blockSignals(False)
        if not saved.autoscale or self.auto is None:
            self.set_levels(saved.min_val, saved.max_val)
        else:
            self.set_levels(*self.auto)
