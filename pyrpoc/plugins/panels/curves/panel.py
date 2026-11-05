"""A loop's recorded intensity against time, one curve per group, updated as it runs: the
group's mean with each unit faint behind it."""

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


def group_means(units: list[UnitView], channel: int) -> tuple[np.ndarray, np.ndarray]:
    """A group's mean curve: records binned in time one period wide, each unit averaged
    within a bin, then the units averaged, at the bin's mean time."""
    # Binned by time rather than by record number, so a unit added mid-run joins the curve
    # where it started instead of at the beginning.
    gaps = [np.diff([t for t, _v in unit.series]) for unit in units if len(unit.series) > 1]
    pooled = np.concatenate(gaps) if gaps else np.array([])
    width = float(np.median(pooled)) if pooled.size else 1.0
    bins: dict[int, dict[int, list[tuple[float, float]]]] = {}
    for unit in units:
        for t, values in unit.series:
            bins.setdefault(int(t // width), {}).setdefault(unit.id, []).append(
                (t, values[channel])
            )
    times, means = [], []
    for index in sorted(bins):
        per_unit = list(bins[index].values())
        times.append(np.mean([t for points in per_unit for t, _v in points]) / 60.0)
        means.append(np.mean([np.mean([v for _t, v in points]) for points in per_unit]))
    return np.asarray(times), np.asarray(means)


@data_panel_registry.register("curves")
class CurvesPanel(DataPanel[LoopLogOutput]):
    display_name = "Curves"
    group = "Autonomous"
    order = 41
    renders = LoopLogOutput

    def __init__(self, library: Library, inputs: ProgramInputs, requests: Requests):
        super().__init__(library, inputs, requests)
        self.view: LoopView | None = None
        root = QVBoxLayout(self.body)
        root.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Channel:", self.body))
        self.channel = QComboBox(self.body)
        self.channel.currentIndexChanged.connect(lambda _index: self.redraw())
        row.addWidget(self.channel)
        row.addStretch(1)
        root.addLayout(row)
        self.plot = pg.PlotWidget(self.body)
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("bottom", "Time (min)")
        self.plot.setLabel("left", "Mean intensity")
        self.legend = self.plot.addLegend()
        root.addWidget(self.plot, 1)
        self.connect_source()

    def draw(self, program_output: LoopLogOutput) -> None:
        self.view = program_output.view()
        labels = list(self.view.channels)
        if [self.channel.itemText(i) for i in range(self.channel.count())] != labels:
            self.channel.blockSignals(True)
            self.channel.clear()
            self.channel.addItems(labels)
            self.channel.blockSignals(False)
        self.redraw()

    def redraw(self) -> None:
        state = self.view
        self.plot.clear()
        if state is None or self.channel.currentIndex() < 0:
            return
        channel = self.channel.currentIndex()
        groups = sorted({unit.group for unit in state.units})
        for index, group in enumerate(groups):
            members = [unit for unit in state.units if unit.group == group and unit.series]
            colour = color_for_index(index + 2)
            for unit in members:
                times = [t / 60.0 for t, _v in unit.series]
                faint = pg.mkPen((*colour, 60), width=1)
                self.plot.plot(times, [v[channel] for _t, v in unit.series], pen=faint)
            times, means = group_means(members, channel)
            name = f"{group or 'ungrouped'} ({len(members)})"
            self.plot.plot(times, means, pen=pg.mkPen(colour, width=2.5), name=name)

    def clear(self) -> None:
        self.view = None
        self.plot.clear()
