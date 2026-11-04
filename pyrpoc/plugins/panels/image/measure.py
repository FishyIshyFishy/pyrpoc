"""Measuring on a tile: drag a line for its profile or a box for a histogram. The result sits
in a slim strip under the tile, and only while a measurement exists."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QPalette
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget
from pyqtgraph.graphicsItems.ViewBox import ViewBox

from . import measurements

STRIP_PLOT_HEIGHT = 96


class Tool(Enum):
    LINE = "line"
    REGION = "region"


HINTS = {Tool.LINE: "Drag to draw a line", Tool.REGION: "Drag to draw a box"}


def accent() -> QColor:
    return QGuiApplication.palette().color(QPalette.ColorRole.Highlight)


def short(value: float) -> str:
    return f"{value:.4g}"


class ReadoutStrip(QFrame):
    """A small plot with one line of numbers beside it, and a button to dismiss it."""

    closed = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
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
        close = QToolButton(self)
        close.setAutoRaise(True)
        close.setText("✕")
        close.setToolTip("Remove this measurement")
        close.clicked.connect(self.closed)
        side.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)
        self.stats = QLabel(self)
        self.stats.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        font = self.stats.font()
        font.setPointSizeF(font.pointSizeF() * 0.9)
        self.stats.setFont(font)
        side.addWidget(self.stats, 1)
        row.addLayout(side)

    def show_histogram(
        self, histogram: measurements.Histogram, stats: measurements.RegionStats
    ) -> None:
        color = accent()
        self.curve.setData(
            histogram.edges,
            histogram.counts,
            stepMode="center",
            fillLevel=0,
            pen=pg.mkPen(color, width=1),
            brush=pg.mkBrush(color.red(), color.green(), color.blue(), 110),
        )
        self.plot.setLabel("bottom", "pixel value")
        self.plot.setLabel("left", "pixels")
        self.plot.enableAutoRange()
        self.stats.setText(
            f"n  {stats.count:,}\nmean  {short(stats.mean)}\nσ  {short(stats.std)}\n"
            f"min  {short(stats.minimum)}\nmax  {short(stats.maximum)}"
        )

    def show_profile(self, profile: measurements.Profile) -> None:
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
        self.plot.enableAutoRange()
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
    """One tile's measurement: its view box, the ROI drawn on it, and the readout under it."""

    def __init__(self, parent: QWidget, active_tool: Callable[[], Tool | None]):
        self.active_tool = active_tool
        self.view_box = ToolViewBox(active_tool, self.on_drag)
        self.strip = ReadoutStrip(parent)
        self.strip.closed.connect(self.clear)
        self.line: pg.LineSegmentROI | None = None
        self.box: pg.RectROI | None = None
        self.plane: np.ndarray | None = None
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

    def on_drag(self, tool: Tool, start: QPointF, here: QPointF, first: bool, last: bool) -> None:
        del last
        if first:
            self.clear()
            self.begin(tool, start)
        if self.line is not None:
            self.line.movePoint(self.line.endpoints[1], here)
        elif self.box is not None:
            self.box.setPos(pg.Point(min(start.x(), here.x()), min(start.y(), here.y())))
            self.box.setSize(pg.Point(abs(here.x() - start.x()), abs(here.y() - start.y())))
        self.measure()

    def begin(self, tool: Tool, start: QPointF) -> None:
        pen = pg.mkPen(accent(), width=2)
        if tool is Tool.LINE:
            self.line = pg.LineSegmentROI([start, start], pen=pen)
            roi = self.line
        else:
            self.box = pg.RectROI(start, [0, 0], pen=pen, invertible=True)
            roi = self.box
        roi.removable = True
        roi.sigRegionChanged.connect(self.measure)
        roi.sigRemoveRequested.connect(self.clear)
        self.view_box.addItem(roi)
        self.hint.setVisible(False)

    def clear(self) -> None:
        for roi in (self.line, self.box):
            if roi is not None:
                self.view_box.removeItem(roi)
        self.line = self.box = None
        self.strip.setVisible(False)
        self.show_hint_for(self.active_tool())

    def refresh(self, plane: np.ndarray) -> None:
        """Re-measure on a new frame, so the readout follows a live acquisition."""
        self.plane = plane
        self.measure()

    def measure(self) -> None:
        if self.plane is None or not self.drawn:
            return
        self.strip.setVisible(True)
        if self.line is not None:
            self.measure_line(self.line, self.plane)
        elif self.box is not None:
            self.measure_box(self.box, self.plane)

    def measure_line(self, line: pg.LineSegmentROI, plane: np.ndarray) -> None:
        ends = [line.mapToParent(pos) for _, pos in line.getLocalHandlePositions()]
        profile = measurements.line_profile(
            plane, (ends[0].x(), ends[0].y()), (ends[1].x(), ends[1].y())
        )
        if profile.values.size == 0:
            self.strip.show_nothing()
        else:
            self.strip.show_profile(profile)

    def measure_box(self, box: pg.RectROI, plane: np.ndarray) -> None:
        origin, size = box.pos(), box.size()
        values = measurements.region_values(
            plane, origin.x(), origin.y(), origin.x() + size.x(), origin.y() + size.y()
        )
        stats = measurements.region_stats(values)
        if stats is None:
            self.strip.show_nothing()
        else:
            self.strip.show_histogram(measurements.histogram(values), stats)
