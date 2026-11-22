"""Every channel overlaid in its own colour, with a LUT per channel."""

# Each channel is one column: a swatch and its name as a show/hide box above its levels control, an
# autoscale box and a vertical histogram filling the rest of the height. The columns share the
# strip's width equally, so a few channels fill it; many scroll sideways.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QScrollArea, QWidget

from pyrpoc.utils.widgets.colors import color_for_index, color_map_from_rgb
from pyrpoc.utils.widgets.levels import LevelsControl, LevelsState, autoscale_levels

# Columns the strip makes room for before scrolling the rest sideways.
COLUMNS_WITHOUT_SCROLLING = 4


@dataclass
class ChannelColumn:
    levels: LevelsControl
    shown: QCheckBox
    rgb: tuple[int, int, int]


@dataclass
class ColumnState:
    """One channel's saved settings, parsed from the session."""

    shown: bool
    levels: LevelsState


class ChannelStrip(QScrollArea):
    """Holds the planes the levels are judged on, one per channel, and colours images of these
    channels with each channel's levels."""

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
        """Judge levels on new data: one ``(H, W)`` plane per channel."""
        self.planes = planes
        self.sync_columns(labels)
        for column, plane in zip(self.columns, planes, strict=True):
            column.levels.take(plane, autoscale_levels(plane))

    def colour(self, planes: np.ndarray) -> np.ndarray:
        """A ``(C, H, W)`` image of these channels as ``(H, W, 3)`` in 0..1: the shown
        channels, each scaled by its levels and tinted its colour."""
        height, width = planes.shape[1:]
        out = np.zeros((height, width, 3), dtype=np.float32)
        for column, plane in zip(self.columns, planes, strict=True):
            if not column.shown.isChecked():
                continue
            scaled = column.levels.scaled(plane)
            out += scaled[..., None] * (np.asarray(column.rgb, dtype=np.float32) / 255.0)
        return np.clip(out, 0.0, 1.0)

    def sync_columns(self, labels: list[str]) -> None:
        """One column per channel, renamed in place; columns that stay keep
        their levels."""
        while len(self.columns) > len(labels):
            column = self.columns.pop()
            column.levels.hide()
            column.levels.deleteLater()
        while len(self.columns) < len(labels):
            self.columns.append(self.build_column(len(self.columns)))
            self._layout.addWidget(self.columns[-1].levels, 1)
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
        needed = shown * (self.columns[0].levels.sizeHint().width() + self._layout.spacing())
        self.setMinimumWidth(needed + margins.left() + margins.right() + 2 * self.frameWidth())

    def build_column(self, index: int) -> ChannelColumn:
        rgb = color_for_index(index)
        # The source image is never drawn: it holds the channel's plane so the histogram has data.
        levels = LevelsControl(self._content, pg.ImageItem(), color_map_from_rgb(rgb))
        swatch = QLabel(levels)
        swatch.setFixedSize(10, 10)
        swatch.setStyleSheet("background-color: rgb({}, {}, {});".format(*rgb))
        levels.header.addWidget(swatch)
        shown = QCheckBox(levels)
        shown.setChecked(True)
        shown.setToolTip("Show this channel")
        levels.header.addWidget(shown, 1)
        shown.toggled.connect(lambda _checked: self.changed.emit())
        levels.changed.connect(self.changed.emit)
        return ChannelColumn(levels, shown, rgb)

    def clear_planes(self) -> None:
        """Nothing is judged; columns stay, keeping their levels for the next data."""
        self.planes = []

    def export_state(self) -> list[dict[str, Any]]:
        return [
            {"shown": column.shown.isChecked(), **column.levels.export_state().to_json()}
            for column in self.columns
        ]

    def import_state(self, raw: Any) -> None:
        """Parse saved channels from session JSON, a boundary: a malformed entry
        is skipped. Applied once columns exist, which is when data first arrives."""
        if not isinstance(raw, list):
            return
        self._pending = [
            ColumnState(shown=bool(entry.get("shown", True)), levels=LevelsState.from_json(entry))
            for entry in raw
            if isinstance(entry, dict)
        ]
        self.apply_pending()

    def apply_pending(self) -> None:
        if not self._pending or not self.columns:
            return
        for column, saved in zip(self.columns, self._pending, strict=False):
            column.shown.blockSignals(True)
            column.shown.setChecked(saved.shown)
            column.shown.blockSignals(False)
            column.levels.restore(saved.levels)
        self._pending = []
