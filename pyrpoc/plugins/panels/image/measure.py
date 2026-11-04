"""Measuring on a tile: drag a line for its profile or a box for a histogram. The result sits
in a slim strip under the tile, and only while a measurement exists."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QPalette
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)
from pyqtgraph.graphicsItems.ViewBox import ViewBox

from . import measurements
from .measure_settings import AxisRange, MeasureSettings, SettingsForm

STRIP_PLOT_HEIGHT = 96


class Tool(Enum):
    LINE = "line"
    REGION = "region"


HINTS = {Tool.LINE: "Drag to draw a line", Tool.REGION: "Drag to draw a box"}


@dataclass(frozen=True)
class Shape:
    """A measuring shape in image pixel units: a line from one point to another, or a box
    between two corners. The same shape is drawn on every tile."""

    tool: Tool
    x0: float
    y0: float
    x1: float
    y1: float


def accent() -> QColor:
    return QGuiApplication.palette().color(QPalette.ColorRole.Highlight)


def count_scale(log_counts: bool) -> Callable[[float], float]:
    """How a count on the y axis is drawn: as is, or as log10(1 + count)."""
    if log_counts:
        return lambda count: float(np.log10(1 + max(count, 0.0)))
    return float


def short(value: float) -> str:
    return f"{value:.4g}"


class ReadoutStrip(QFrame):
    """A small plot with one line of numbers beside it, and a button to dismiss it."""

    closed = pyqtSignal()

    def __init__(self, settings: MeasureSettings, parent: QWidget):
        super().__init__(parent)
        self.settings = settings
        self.setVisible(False)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.plot = pg.PlotWidget(self)
        self.plot.setFixedHeight(STRIP_PLOT_HEIGHT)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(False, False)
        self.plot.hideButtons()
        self.plot.getAxis("left").setWidth(42)
        self.curve = pg.PlotCurveItem()
        self.plot.addItem(self.curve)
        row.addWidget(self.plot, 1)

        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        buttons = QHBoxLayout()
        buttons.setSpacing(0)
        buttons.addStretch(1)
        self.form = SettingsForm(settings, self)
        buttons.addWidget(self.build_settings_button())
        close = QToolButton(self)
        close.setAutoRaise(True)
        close.setText("✕")
        close.setToolTip("Remove this measurement")
        close.clicked.connect(self.closed)
        buttons.addWidget(close)
        side.addLayout(buttons)
        self.stats = QLabel(self)
        self.stats.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        font = self.stats.font()
        font.setPointSizeF(font.pointSizeF() * 0.9)
        self.stats.setFont(font)
        side.addWidget(self.stats, 1)
        row.addLayout(side)

    def build_settings_button(self) -> QToolButton:
        button = QToolButton(self)
        button.setAutoRaise(True)
        button.setText("⚙")
        button.setToolTip("Plot settings: autoscaling, ranges and bins")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        menu = QMenu(button)
        action = QWidgetAction(menu)
        action.setDefaultWidget(self.form)
        menu.addAction(action)
        button.setMenu(menu)
        return button

    def apply_range(
        self, axis: int, axis_range: AxisRange, transform: Callable[[float], float]
    ) -> None:
        """Fit ``axis`` to the data, or hold it at the chosen range when that is a real one."""
        if axis_range.auto or axis_range.high <= axis_range.low:
            self.plot.enableAutoRange(axis=axis)
        elif axis == ViewBox.XAxis:
            self.plot.getViewBox().setXRange(axis_range.low, axis_range.high, padding=0)
        else:
            self.plot.getViewBox().setYRange(
                transform(axis_range.low), transform(axis_range.high), padding=0
            )

    def show_histogram(
        self, histogram: measurements.Histogram, stats: measurements.RegionStats
    ) -> None:
        self.form.show_profile(False)
        settings = self.settings
        counts = np.log10(1 + histogram.counts) if settings.log_counts else histogram.counts
        color = accent()
        self.curve.setData(
            histogram.edges,
            counts,
            stepMode="center",
            fillLevel=0,
            pen=pg.mkPen(color, width=1),
            brush=pg.mkBrush(color.red(), color.green(), color.blue(), 110),
        )
        self.plot.setLabel("bottom", "pixel value")
        self.plot.setLabel("left", "log10(1 + pixels)" if settings.log_counts else "pixels")
        self.apply_range(ViewBox.XAxis, settings.hist_x, float)
        self.apply_range(ViewBox.YAxis, settings.hist_y, count_scale(settings.log_counts))
        self.stats.setText(
            f"n  {stats.count:,}\nmean  {short(stats.mean)}\nσ  {short(stats.std)}\n"
            f"min  {short(stats.minimum)}\nmax  {short(stats.maximum)}"
        )

    def show_profile(self, profile: measurements.Profile) -> None:
        self.form.show_profile(True)
        self.curve.setData(
            profile.distance,
            profile.values,
            stepMode=None,
            fillLevel=None,
            brush=None,
            pen=pg.mkPen(accent(), width=2),
        )
        self.plot.setLabel("bottom", "distance (px)")
        self.plot.setLabel("left", "value")
        self.plot.enableAutoRange(axis=ViewBox.XAxis)
        self.apply_range(ViewBox.YAxis, self.settings.profile_y, float)
        values = profile.values
        self.stats.setText(
            f"length  {short(profile.length)} px\nmean  {short(float(values.mean()))}\n"
            f"min  {short(float(values.min()))}\nmax  {short(float(values.max()))}"
        )

    def show_nothing(self) -> None:
        self.curve.setData([], [])
        self.stats.setText("outside the image")


class ToolViewBox(ViewBox):
    """A view box whose left-drag draws while a tool is active, and pans otherwise."""

    def __init__(
        self,
        active_tool: Callable[[], Tool | None],
        dragged: Callable[[Tool, QPointF, QPointF, bool, bool], None],
    ):
        super().__init__()
        self.active_tool = active_tool
        self.dragged = dragged

    def mouseDragEvent(self, ev: Any, axis: Any = None) -> None:
        tool = self.active_tool()
        if tool is None or ev.button() != Qt.MouseButton.LeftButton:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        start = self.mapToView(ev.buttonDownPos())
        here = self.mapToView(ev.pos())
        self.dragged(tool, start, here, ev.isStart(), ev.isFinish())


class TileMeasurement:
    """One tile's measurement: its view box, the ROI drawn on it, and the readout under it.
    A change the user makes is reported through ``on_shape`` so the other tiles can follow."""

    def __init__(
        self,
        parent: QWidget,
        settings: MeasureSettings,
        active_tool: Callable[[], Tool | None],
        on_shape: Callable[[TileMeasurement, Shape | None], None],
    ):
        self.active_tool = active_tool
        self.on_shape = on_shape
        self.view_box = ToolViewBox(active_tool, self.on_drag)
        self.strip = ReadoutStrip(settings, parent)
        self.strip.closed.connect(self.remove)
        settings.changed.connect(self.measure)
        self.line: pg.LineSegmentROI | None = None
        self.box: pg.RectROI | None = None
        self.plane: np.ndarray | None = None
        # Set while a shape is being applied, so the ROI's own change signals are not
        # mistaken for the user's edits.
        self.applying = False
        self.hint = pg.TextItem(anchor=(0, 0), color=(255, 255, 255), fill=(0, 0, 0, 150))
        self.hint.setVisible(False)
        # Above the image, which is added after it.
        self.hint.setZValue(10)
        self.view_box.addItem(self.hint, ignoreBounds=True)

    @property
    def drawn(self) -> bool:
        return self.line is not None or self.box is not None

    def show_hint_for(self, tool: Tool | None) -> None:
        """Say how to start while a tool is active and nothing is drawn yet."""
        if tool is not None:
            self.hint.setText(HINTS[tool])
        self.hint.setVisible(tool is not None and not self.drawn)

    def shape(self) -> Shape | None:
        if self.line is not None:
            ends = [self.line.mapToParent(pos) for _, pos in self.line.getLocalHandlePositions()]
            return Shape(Tool.LINE, ends[0].x(), ends[0].y(), ends[1].x(), ends[1].y())
        if self.box is not None:
            origin, size = self.box.pos(), self.box.size()
            return Shape(
                Tool.REGION, origin.x(), origin.y(), origin.x() + size.x(), origin.y() + size.y()
            )
        return None

    def apply_shape(self, shape: Shape | None) -> None:
        """Show ``shape`` here, replacing whatever is drawn, without reporting it back."""
        self.applying = True
        try:
            if shape is None:
                self.remove_rois()
            else:
                self.place(shape)
        finally:
            self.applying = False
        self.hint.setVisible(False)
        self.show_hint_for(self.active_tool())
        self.measure()

    def place(self, shape: Shape) -> None:
        wanted = self.line if shape.tool is Tool.LINE else self.box
        if wanted is None:
            self.remove_rois()
            self.begin(shape.tool, QPointF(shape.x0, shape.y0))
        self.set_geometry(shape)

    def set_geometry(self, shape: Shape) -> None:
        if self.line is not None:
            self.line.movePoint(self.line.endpoints[0], QPointF(shape.x0, shape.y0), finish=False)
            self.line.movePoint(self.line.endpoints[1], QPointF(shape.x1, shape.y1), finish=False)
        elif self.box is not None:
            self.box.setPos(pg.Point(min(shape.x0, shape.x1), min(shape.y0, shape.y1)))
            self.box.setSize(pg.Point(abs(shape.x1 - shape.x0), abs(shape.y1 - shape.y0)))

    def on_drag(self, tool: Tool, start: QPointF, here: QPointF, first: bool, last: bool) -> None:
        del last
        if first:
            self.apply_shape(None)
        self.apply_shape(Shape(tool, start.x(), start.y(), here.x(), here.y()))
        self.on_shape(self, self.shape())

    def begin(self, tool: Tool, start: QPointF) -> None:
        pen = pg.mkPen(accent(), width=2)
        if tool is Tool.LINE:
            self.line = pg.LineSegmentROI([start, start], pen=pen)
            roi = self.line
        else:
            self.box = pg.RectROI(start, [0, 0], pen=pen, invertible=True)
            roi = self.box
        roi.removable = True
        roi.sigRegionChanged.connect(self.on_roi_changed)
        roi.sigRemoveRequested.connect(self.remove)
        self.view_box.addItem(roi)

    def on_roi_changed(self) -> None:
        if self.applying:
            return
        self.measure()
        self.on_shape(self, self.shape())

    def remove_rois(self) -> None:
        for roi in (self.line, self.box):
            if roi is not None:
                self.view_box.removeItem(roi)
        self.line = self.box = None
        self.strip.setVisible(False)

    def remove(self) -> None:
        """The user's removal of the measurement, which every tile follows."""
        self.apply_shape(None)
        self.on_shape(self, None)

    def refresh(self, plane: np.ndarray) -> None:
        """Re-measure on a new frame, so the readout follows a live acquisition."""
        self.plane = plane
        self.measure()

    def measure(self) -> None:
        shape = self.shape()
        if self.plane is None or shape is None:
            return
        self.strip.setVisible(True)
        if shape.tool is Tool.LINE:
            profile = measurements.line_profile(
                self.plane, (shape.x0, shape.y0), (shape.x1, shape.y1)
            )
            if profile.values.size == 0:
                self.strip.show_nothing()
            else:
                self.strip.show_profile(profile)
        else:
            values = measurements.region_values(self.plane, shape.x0, shape.y0, shape.x1, shape.y1)
            stats = measurements.region_stats(values)
            if stats is None:
                self.strip.show_nothing()
            else:
                settings = self.strip.settings
                bins = None if settings.auto_bins else settings.bins
                self.strip.show_histogram(measurements.histogram(values, bins), stats)
