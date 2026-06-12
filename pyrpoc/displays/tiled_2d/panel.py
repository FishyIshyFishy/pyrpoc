"""Results panel for the analysis tools (readout / profile plot / stats table)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .statistics import RegionStats
from .tile import channel_color
from .toolbar import AnalysisTool


stats_columns = ["Channel", "N", "Mean", "Std", "Min", "Max", "Sum"]


def format_number(value: float) -> str:
    return f"{value:.4g}"


class AnalysisPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)

        self._title = QLabel("", self)
        self._title.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._title)

        self._stack = QStackedWidget(self)
        layout.addWidget(self._stack, 1)

        self._readout = QLabel("Move the cursor over an image.", self)
        self._readout.setWordWrap(True)
        self._stack.addWidget(self._readout)

        self._profile_plot = pg.PlotWidget(self)
        self._profile_plot.setMenuEnabled(False)
        self._profile_plot.setLabel("bottom", "Distance (px)")
        self._profile_plot.setLabel("left", "Intensity")
        self._legend = self._profile_plot.addLegend()
        self._profile_curves: list[pg.PlotDataItem] = []
        self._stack.addWidget(self._profile_plot)

        self._table = QTableWidget(self)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setColumnCount(len(stats_columns))
        self._table.setHorizontalHeaderLabels(stats_columns)
        vertical_header = self._table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._stack.addWidget(self._table)

    def show_tool(self, tool: AnalysisTool) -> None:
        if tool == AnalysisTool.INSPECT:
            self._title.setText("Pixel inspector")
            self._stack.setCurrentWidget(self._readout)
            self._readout.setText("Move the cursor over an image.")
        elif tool == AnalysisTool.LINE:
            self._title.setText("Line profile")
            self._stack.setCurrentWidget(self._profile_plot)
        elif tool in (AnalysisTool.RECTANGLE, AnalysisTool.ELLIPSE):
            self._title.setText("Region statistics")
            self._stack.setCurrentWidget(self._table)
        else:
            self._title.setText("")
            self._stack.setCurrentWidget(self._readout)

    def show_readout(self, x: int | None, y: int | None, names: list[str], values: list[float | None]) -> None:
        if x is None or y is None:
            self._readout.setText("Cursor outside image.")
            return
        lines = [f"x = {x}    y = {y}", ""]
        for name, value in zip(names, values):
            text = "—" if value is None else format_number(value)
            lines.append(f"{name}:  {text}")
        self._readout.setText("\n".join(lines))

    def show_profiles(self, distances: np.ndarray, names: list[str], profiles: list[np.ndarray]) -> None:
        for curve in self._profile_curves:
            self._profile_plot.removeItem(curve)
        self._profile_curves = []
        if self._legend is not None:
            self._legend.clear()
        for index, (name, profile) in enumerate(zip(names, profiles)):
            pen = pg.mkPen(color=channel_color(index), width=2)
            curve = self._profile_plot.plot(distances, profile, pen=pen, name=name)
            self._profile_curves.append(curve)

    def show_stats(self, names: list[str], stats: list[RegionStats]) -> None:
        self._table.setRowCount(len(names))
        for row, (name, stat) in enumerate(zip(names, stats)):
            cells = [
                name,
                str(stat.count),
                format_number(stat.mean),
                format_number(stat.std),
                format_number(stat.minimum),
                format_number(stat.maximum),
                format_number(stat.total),
            ]
            for col, text in enumerate(cells):
                self._table.setItem(row, col, QTableWidgetItem(text))
