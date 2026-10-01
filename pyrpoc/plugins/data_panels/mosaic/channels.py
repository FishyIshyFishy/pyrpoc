"""Every channel overlaid in its own colour, with a LUT per channel.

Each channel is one column: a swatch and its name as a show/hide box, an
autoscale box, and a vertical histogram filling the rest of the height. The
columns share the strip's width equally, so a few channels fill it without
leaving space over; many scroll sideways.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from pyrpoc.qt_components.colors import color_for_index, color_map_from_rgb
from pyrpoc.qt_components.levels import autoscale_levels, mono_levels

# Columns the strip makes room for before scrolling the rest sideways.
COLUMNS_WITHOUT_SCROLLING = 4


@dataclass
class ChannelColumn:
    root: QWidget
    shown: QCheckBox
    autoscale: QCheckBox
    histogram: pg.HistogramLUTWidget
    # Never drawn: it holds the channel's plane so the histogram has data.
    source: pg.ImageItem
    rgb: tuple[int, int, int]


@dataclass
class ColumnState:
    """One channel's saved settings, parsed from the session."""

    shown: bool
    autoscale: bool
    min_val: float
    max_val: float


class ChannelStrip(QScrollArea):
    """Holds the planes being shown, one per channel, and colours them into
    one RGB image with each channel's levels."""

    # The user changed a channel's levels, visibility or autoscale.
    changed = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.columns: list[ChannelColumn] = []
        self.planes: list[np.ndarray] = []
        self._pending: list[ColumnState] = []
        self._content = QWidget(self)
        self._layout = QHBoxLayout(self._content)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        self.setWidget(self._content)

    def set_planes(self, planes: list[np.ndarray], labels: list[str]) -> None:
        """Show new data: one ``(H, W)`` plane per channel."""
        self.planes = planes
        self.sync_columns(labels)
        for column, plane in zip(self.columns, planes, strict=True):
            column.source.setImage(plane, autoLevels=False)
            if column.autoscale.isChecked():
                self.apply_levels(column, *autoscale_levels(plane))

    def rgb(self) -> np.ndarray:
        """The shown channels, each scaled by its levels and tinted its colour."""
        height, width = self.planes[0].shape
        out = np.zeros((height, width, 3), dtype=np.float32)
        for column, plane in zip(self.columns, self.planes, strict=True):
            if not column.shown.isChecked():
                continue
            low, high = mono_levels(column.histogram)
            scaled = np.clip((plane - low) / max(high - low, 1e-12), 0.0, 1.0)
            out += scaled[..., None] * (np.asarray(column.rgb, dtype=np.float32) / 255.0)
        return np.clip(out, 0.0, 1.0)

    def apply_levels(self, column: ChannelColumn, low: float, high: float) -> None:
        # Blocked so levels set here are not mistaken for the user dragging them.
        column.histogram.item.blockSignals(True)
        column.histogram.item.setLevels(low, max(high, low + 1e-12))
        column.histogram.item.blockSignals(False)

    def sync_columns(self, labels: list[str]) -> None:
        """One column per channel, renamed in place; columns that stay keep
        their levels."""
        while len(self.columns) > len(labels):
            column = self.columns.pop()
            column.root.hide()
            column.root.deleteLater()
        while len(self.columns) < len(labels):
            self.columns.append(self.build_column(len(self.columns)))
            self._layout.addWidget(self.columns[-1].root, 1)
        for column, label in zip(self.columns, labels, strict=True):
            column.shown.setText(label)
        self.fit_columns()
        self.apply_pending()

    def fit_columns(self) -> None:
        """Ask the splitter for room to show every column, up to a few; a
        scroll area does not pass on the width its contents need."""
        shown = min(len(self.columns), COLUMNS_WITHOUT_SCROLLING)
        if shown == 0:
            return
        margins = self._layout.contentsMargins()
        needed = shown * (self.columns[0].root.sizeHint().width() + self._layout.spacing())
        self.setMinimumWidth(needed + margins.left() + margins.right() + 2 * self.frameWidth())

    def build_column(self, index: int) -> ChannelColumn:
        rgb = color_for_index(index)
        root = QWidget(self._content)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        name = QHBoxLayout()
        swatch = QLabel(root)
        swatch.setFixedSize(10, 10)
        swatch.setStyleSheet("background-color: rgb({}, {}, {});".format(*rgb))
        name.addWidget(swatch)
        shown = QCheckBox(root)
        shown.setChecked(True)
        shown.setToolTip("Show this channel")
        name.addWidget(shown, 1)
        layout.addLayout(name)
        autoscale = QCheckBox("Auto", root)
        autoscale.setChecked(True)
        autoscale.setToolTip("Rescale this channel's levels to each new image")
        layout.addWidget(autoscale)

        source = pg.ImageItem()
        histogram = pg.HistogramLUTWidget(root)
        histogram.setImageItem(source)
        histogram.item.gradient.setColorMap(color_map_from_rgb(rgb))
        layout.addWidget(histogram, 1)

        column = ChannelColumn(root, shown, autoscale, histogram, source, rgb)
        shown.toggled.connect(lambda _checked: self.changed.emit())
        autoscale.toggled.connect(lambda _checked, c=column: self.on_autoscale_toggled(c))
        histogram.item.sigLevelsChanged.connect(lambda _item: self.changed.emit())
        return column

    def on_autoscale_toggled(self, column: ChannelColumn) -> None:
        if column.autoscale.isChecked() and self.planes:
            plane = self.planes[self.columns.index(column)]
            self.apply_levels(column, *autoscale_levels(plane))
        self.changed.emit()

    def clear_planes(self) -> None:
        """Nothing is shown; columns stay, keeping their levels for the next data."""
        self.planes = []

    def export_state(self) -> list[dict[str, Any]]:
        state = []
        for column in self.columns:
            low, high = mono_levels(column.histogram)
            state.append(
                {
                    "shown": column.shown.isChecked(),
                    "autoscale": column.autoscale.isChecked(),
                    "min_val": low,
                    "max_val": high,
                }
            )
        return state

    def import_state(self, raw: Any) -> None:
        """Parse saved channels from session JSON, a boundary: a malformed entry
        is skipped. Applied once columns exist, which is when data first arrives."""
        if not isinstance(raw, list):
            return
        self._pending = [
            ColumnState(
                shown=bool(entry.get("shown", True)),
                autoscale=bool(entry.get("autoscale", True)),
                min_val=float(entry.get("min_val", 0.0)),
                max_val=float(entry.get("max_val", 1.0)),
            )
            for entry in raw
            if isinstance(entry, dict)
        ]
        self.apply_pending()

    def apply_pending(self) -> None:
        if not self._pending or not self.columns:
            return
        for column, saved in zip(self.columns, self._pending, strict=False):
            for box, value in ((column.shown, saved.shown), (column.autoscale, saved.autoscale)):
                box.blockSignals(True)
                box.setChecked(value)
                box.blockSignals(False)
            self.apply_levels(column, saved.min_val, saved.max_val)
        self._pending = []
