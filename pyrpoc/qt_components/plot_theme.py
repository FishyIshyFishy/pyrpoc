"""Keep pyqtgraph widgets on the application palette.

pyqtgraph fixes its colors when a widget is built, so a theme switch leaves
plots black with gray axes. A widget receives a palette change when the
application palette is set, which is the cue to restyle.
"""

from __future__ import annotations

from collections.abc import Callable

import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtGui import QPalette


class _PaletteFollower(QObject):
    def __init__(self, widget: pg.GraphicsView, restyle: Callable[[], None]) -> None:
        super().__init__(widget)
        self._restyle = restyle
        widget.installEventFilter(self)
        restyle()

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is not None and a1.type() == QEvent.Type.PaletteChange:
            self._restyle()
        return False


def follow_plot(plot: pg.PlotWidget) -> None:
    """Background, axes and grid text of ``plot`` track the palette."""

    def restyle() -> None:
        palette = plot.palette()
        plot.setBackground(palette.color(QPalette.ColorRole.Base))
        text = palette.color(QPalette.ColorRole.Text)
        muted = palette.color(QPalette.ColorRole.Mid)
        plot_item = plot.getPlotItem()
        if plot_item is None:
            raise RuntimeError("a new PlotWidget has no PlotItem")
        for name in ("left", "bottom", "top", "right"):
            axis = plot_item.getAxis(name)
            axis.setPen(muted)
            axis.setTextPen(text)

    _PaletteFollower(plot, restyle)


def follow_histogram(histogram: pg.HistogramLUTWidget) -> None:
    """Background and axis of the levels widget track the palette."""

    def restyle() -> None:
        palette = histogram.palette()
        histogram.setBackground(palette.color(QPalette.ColorRole.Base))
        axis = histogram.item.axis
        axis.setPen(palette.color(QPalette.ColorRole.Mid))
        axis.setTextPen(palette.color(QPalette.ColorRole.Text))

    _PaletteFollower(histogram, restyle)
