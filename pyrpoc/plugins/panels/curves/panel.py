"""A loop's measurements against time, one curve per group, updated as it runs: the group's
mean with each unit faint behind it."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout

from pyrpoc.core.library import Library
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.core.requests import Requests
from pyrpoc.plugins.program_io.outputs import LoopLogOutput, LoopView, UnitView
from pyrpoc.utils.widgets.colors import color_for_index
from pyrpoc.utils.widgets.data_panel import DataPanel, data_panel_registry

Points = tuple[tuple[float, tuple[float, ...]], ...]


def points_of(unit: UnitView, routine: str, name: str) -> Points:
    """Every value ``routine`` recorded as ``name`` of ``unit``, empty if it recorded none."""
    wanted = (routine, name)
    return next((each.points for each in unit.series if (each.routine, each.name) == wanted), ())


def measurements(view: LoopView) -> list[tuple[str, str]]:
    """Every (routine, name) any unit has a series of, in the order first seen."""
    seen: dict[tuple[str, str], None] = {}
    for unit in view.units:
        for series in unit.series:
            seen.setdefault((series.routine, series.name), None)
    return list(seen)


def measurement_labels(series: list[tuple[str, str]]) -> list[str]:
    """Each measurement by its name, with its routine only where two routines share a name."""
    names = [name for _routine, name in series]
    return [name if names.count(name) == 1 else f"{name} ({routine})" for routine, name in series]


def group_means(curves: list[Points], channel: int) -> tuple[np.ndarray, np.ndarray]:
    """A group's mean curve: points binned in time one period wide, each unit averaged
    within a bin, then the units averaged, at the bin's mean time in minutes."""
    # Binned by time rather than by point number, so a unit added mid-run joins the curve
    # where it started instead of at the beginning.
    gaps = [np.diff([t for t, _v in points]) for points in curves if len(points) > 1]
    pooled = np.concatenate(gaps) if gaps else np.array([])
    width = float(np.median(pooled)) if pooled.size else 1.0
    bins: dict[int, dict[int, list[tuple[float, float]]]] = {}
    for index, points in enumerate(curves):
        for t, values in points:
            bins.setdefault(int(t // width), {}).setdefault(index, []).append((t, values[channel]))
    times, means = [], []
    for key in sorted(bins):
        per_unit = list(bins[key].values())
        times.append(np.mean([t for each in per_unit for t, _v in each]) / 60.0)
        means.append(np.mean([np.mean([v for _t, v in each]) for each in per_unit]))
    return np.asarray(times), np.asarray(means)


def set_items(combo: QComboBox, labels: list[str]) -> None:
    """Give ``combo`` these labels, keeping its choice, without a change signal when it already
    has them."""
    if [combo.itemText(i) for i in range(combo.count())] == labels:
        return
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(labels)
    combo.blockSignals(False)


@data_panel_registry.register("curves")
class CurvesPanel(DataPanel[LoopLogOutput]):
    display_name = "Site Curves"
    group = "Multisite"
    order = 41
    renders = LoopLogOutput

    def __init__(self, library: Library, inputs: ProgramInputs, requests: Requests):
        super().__init__(library, inputs, requests)
        self.view: LoopView | None = None
        self.series: list[tuple[str, str]] = []
        root = QVBoxLayout(self.body)
        root.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Measurement:", self.body))
        self.measurement = QComboBox(self.body)
        self.measurement.currentIndexChanged.connect(lambda _index: self.redraw())
        row.addWidget(self.measurement)
        row.addWidget(QLabel("Channel:", self.body))
        self.channel = QComboBox(self.body)
        self.channel.currentIndexChanged.connect(lambda _index: self.redraw())
        row.addWidget(self.channel)
        row.addStretch(1)
        root.addLayout(row)
        self.plot = pg.PlotWidget(self.body)
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("bottom", "Time (min)")
        self.legend = self.plot.addLegend()
        root.addWidget(self.plot, 1)
        self.connect_source()

    def draw(self, program_output: LoopLogOutput) -> None:
        self.view = program_output.view()
        self.series = measurements(self.view)
        set_items(self.measurement, measurement_labels(self.series))
        set_items(self.channel, list(self.view.channels))
        self.redraw()

    def redraw(self) -> None:
        view = self.view
        self.plot.clear()
        chosen, channel = self.measurement.currentIndex(), self.channel.currentIndex()
        if view is None or chosen < 0 or channel < 0:
            return
        routine, name = self.series[chosen]
        self.plot.setLabel("left", name)
        groups = sorted({unit.group for unit in view.units})
        for index, group in enumerate(groups):
            curves = [points_of(unit, routine, name) for unit in view.units if unit.group == group]
            self.plot_group(group, [points for points in curves if points], channel, index)

    def plot_group(self, group: str, curves: list[Points], channel: int, index: int) -> None:
        """Each unit's curve faint, and the group's mean over them."""
        if not curves:
            return
        colour = color_for_index(index + 2)
        faint = pg.mkPen((*colour, 60), width=1)
        for points in curves:
            minutes = [t / 60.0 for t, _v in points]
            self.plot.plot(minutes, [v[channel] for _t, v in points], pen=faint)
        times, means = group_means(curves, channel)
        label = f"{group or 'no group'} ({len(curves)})"
        self.plot.plot(times, means, pen=pg.mkPen(colour, width=2.5), name=label)

    def clear(self) -> None:
        self.view = None
        self.plot.clear()
