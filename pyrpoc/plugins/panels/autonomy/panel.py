"""Units drawn over a survey's overview: draw, move, reshape, group and delete them, and while
a loop runs on them, watch and steer it. Units are the ``UNITS`` program input, kept in
``units.json`` in the survey's folder."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget

from pyrpoc.core.library import Library
from pyrpoc.core.program_input import ParameterError, ProgramInputs
from pyrpoc.core.requests import Requests
from pyrpoc.plugins.program_io.inputs import PAUSED_UNITS, UNITS, Unit
from pyrpoc.plugins.program_io.outputs import (
    LoopLogOutput,
    LoopView,
    PlacedImageOutput,
    SurveyOutput,
)
from pyrpoc.plugins.program_io.requests import RUN_NOW
from pyrpoc.utils.widgets.data_panel import DataPanel, data_panel_registry

from .canvas import UnitCanvas
from .loop_watch import LoopWatch
from .picture import backdrop
from .units_file import read_units, write_units
from .units_table import UnitsTable

# How often the newest loop's log is looked at again: the loop is not this panel's output, so
# nothing tells the panel when it changes.
WATCH_MS = 500


@data_panel_registry.register("autonomy")
class AutonomyPanel(DataPanel[SurveyOutput]):
    display_name = "Autonomy"
    group = "Autonomous"
    order = 40
    renders = SurveyOutput

    def __init__(self, library: Library, inputs: ProgramInputs, requests: Requests):
        super().__init__(library, inputs, requests)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.folder: Path | None = None
        self.chosen: int | None = None
        # What is drawn, so a redraw with nothing new does no work.
        self.backdrop_of: tuple[PlacedImageOutput, int] | None = None
        self.watched: tuple[LoopLogOutput, int, bool] | None = None
        self.loop: LoopLogOutput | None = None
        self.loop_view: LoopView | None = None

        root = QVBoxLayout(self.body)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self.build_tools())
        root.addWidget(self.build_splitter(), 1)
        self.status = QLabel("", self.body)
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.connect_signals()
        self.watch = QTimer(self)
        self.watch.setInterval(WATCH_MS)
        self.watch.timeout.connect(self.follow_loop)
        self.watch.start()
        self.connect_source()
        self.show_units()

    def build_tools(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.draw_button = QPushButton("Draw unit", self.body)
        self.draw_button.setCheckable(True)
        self.draw_button.setToolTip(
            "Click each corner; double-click, or click the first corner, to close it"
        )
        row.addWidget(self.draw_button)
        self.delete_button = QPushButton("Delete", self.body)
        row.addWidget(self.delete_button)
        row.addStretch(1)
        return row

    def build_splitter(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal, self.body)
        splitter.setChildrenCollapsible(False)
        self.canvas = UnitCanvas(splitter)
        side = QWidget(splitter)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = UnitsTable(side)
        layout.addWidget(self.table, 1)
        # Only while there is a loop to watch: drawing units needs none of it.
        self.loop_watch = LoopWatch(side)
        self.loop_watch.setVisible(False)
        layout.addWidget(self.loop_watch, 1)
        splitter.addWidget(self.canvas)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([700, 300])
        return splitter

    def connect_signals(self) -> None:
        self.draw_button.toggled.connect(self.on_draw_toggled)
        self.delete_button.clicked.connect(self.delete_chosen)
        self.canvas.drawn.connect(self.on_drawn)
        self.canvas.reshaped.connect(self.on_reshaped)
        self.canvas.selected.connect(self.choose)
        self.table.chosen.connect(self.choose)
        self.table.regrouped.connect(self.on_regrouped)
        self.loop_watch.run_now.connect(self.ask_run_now)
        self.loop_watch.pause.connect(self.toggle_pause)
        self.inputs.subscribe(self.on_input_changed)
        self.requests.subscribe(self.show_units)

    @property
    def units(self) -> tuple[Unit, ...]:
        return self.inputs.get(UNITS.key)

    def draw(self, program_output: SurveyOutput) -> None:
        self.show_backdrop(program_output)
        folder = program_output.recording.folder
        # Only on a new folder, so a redraw never undoes units changed since.
        if folder != self.folder:
            self.folder = folder
            self.load_units()

    def show_backdrop(self, survey: SurveyOutput) -> None:
        """The survey's overview, once it has stitched one."""
        overviews = [
            output
            for output in survey.recording.outputs.values()
            if isinstance(output, PlacedImageOutput)
        ]
        if not overviews or not len(overviews[0]):
            return
        overview = overviews[0]
        if self.backdrop_of == (overview, len(overview)):
            return
        self.backdrop_of = (overview, len(overview))
        rgb, rect = backdrop(*overview.latest())
        self.canvas.show_backdrop(rgb, rect)

    def load_units(self) -> None:
        """Take the units kept in the survey's folder."""
        if self.folder is None:
            self.status.setText("This survey is not saved, so units drawn on it will not be kept")
            return
        try:
            units = read_units(self.folder)
        except (OSError, ValueError, KeyError, ParameterError) as exc:
            self.status.setText(f"Could not read the units in this folder: {exc}")
            return
        self.inputs.set(UNITS.key, units)

    def set_units(self, units: tuple[Unit, ...]) -> None:
        """Make ``units`` the program input, and keep them in the survey's folder."""
        self.inputs.set(UNITS.key, units)
        if self.folder is not None:
            write_units(self.folder, units)

    def on_input_changed(self, key: str) -> None:
        if key in (UNITS.key, PAUSED_UNITS.key):
            self.show_units()

    def show_units(self) -> None:
        units = self.units
        if self.chosen not in {unit.id for unit in units}:
            self.chosen = None
        view = self.loop_view
        states = {} if view is None else {unit.id: unit for unit in view.units}
        self.canvas.show_units(units, self.chosen)
        self.table.show_units(units, self.chosen, states, 0.0 if view is None else view.elapsed_s)
        self.delete_button.setEnabled(self.chosen is not None)
        running = self.loop is not None and not self.loop.recording.finished
        paused = self.chosen in self.inputs.get(PAUSED_UNITS.key)
        accepted = RUN_NOW in self.requests.accepted()
        self.loop_watch.set_unit(self.chosen is not None, paused, running, accepted)
        if self.folder is not None:
            self.status.setText(f"{len(units)} units, kept in this survey's folder")

    def follow_loop(self) -> None:
        """Show what the newest loop is doing, while there is one."""
        loops = [
            output
            for recording in self.library.recordings()
            for output in recording.outputs.values()
            if isinstance(output, LoopLogOutput)
        ]
        self.loop = loops[-1] if loops else None
        if self.shown_output is not None:
            self.show_backdrop(self.shown_output)
        if self.loop is None:
            self.forget_loop()
            return
        seen = (self.loop, len(self.loop.events), self.loop.recording.finished)
        if seen == self.watched:
            return
        self.watched = seen
        view = self.loop_view = self.loop.view()
        self.loop_watch.setVisible(True)
        self.loop_watch.show_view(view, not self.loop.recording.finished)
        self.canvas.show_tracked([unit.outline for unit in view.units if unit.state != "done"])
        self.show_units()

    def forget_loop(self) -> None:
        if self.watched is None:
            return
        self.watched = self.loop_view = None
        self.loop_watch.setVisible(False)
        self.canvas.show_tracked([])
        self.show_units()

    def choose(self, unit_id: int | None) -> None:
        if unit_id != self.chosen:
            self.chosen = unit_id
            self.show_units()

    def ask_run_now(self) -> None:
        if self.chosen is not None and not self.requests.post(RUN_NOW, self.chosen):
            self.status.setText("No running loop takes Run now")

    def toggle_pause(self) -> None:
        if self.chosen is None:
            return
        paused: tuple[int, ...] = self.inputs.get(PAUSED_UNITS.key)
        if self.chosen in paused:
            self.inputs.set(PAUSED_UNITS.key, tuple(u for u in paused if u != self.chosen))
        else:
            self.inputs.set(PAUSED_UNITS.key, (*paused, self.chosen))

    def on_draw_toggled(self, on: bool) -> None:
        if on:
            self.canvas.start_drawing()
        else:
            self.canvas.stop_drawing()

    def on_drawn(self, outline: tuple[tuple[float, float], ...]) -> None:
        units = self.units
        new = Unit(max((unit.id for unit in units), default=0) + 1, outline, "")
        self.chosen = new.id
        self.set_units((*units, new))

    def on_reshaped(self, unit_id: int, outline: tuple[tuple[float, float], ...]) -> None:
        self.set_units(
            tuple(replace(u, outline=outline) if u.id == unit_id else u for u in self.units)
        )

    def on_regrouped(self, unit_id: int, group: str) -> None:
        self.set_units(tuple(replace(u, group=group) if u.id == unit_id else u for u in self.units))

    def delete_chosen(self) -> None:
        chosen = self.chosen
        self.chosen = None
        self.set_units(tuple(unit for unit in self.units if unit.id != chosen))

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is not None and event.key() == Qt.Key.Key_Escape and self.draw_button.isChecked():
            self.draw_button.setChecked(False)
            return
        if event is not None and event.key() == Qt.Key.Key_Delete and self.chosen is not None:
            self.delete_chosen()
            return
        super().keyPressEvent(event)

    def clear(self) -> None:
        self.backdrop_of = None
        self.canvas.clear_image()

    def detach(self) -> None:
        self.watch.stop()
        self.inputs.unsubscribe(self.on_input_changed)
        self.requests.unsubscribe(self.show_units)
        super().detach()
