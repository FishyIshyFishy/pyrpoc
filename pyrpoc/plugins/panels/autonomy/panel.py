"""Units drawn over a survey's overview: draw, move, reshape, group and delete them, and while
a loop runs on them, watch and steer it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget

from pyrpoc.core.library import Library
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.requests import Requests
from pyrpoc.plugins.program_io.inputs import UNITS
from pyrpoc.plugins.program_io.outputs import (
    LoopLogOutput,
    LoopView,
    Outline,
    SurveyOutput,
    Unit,
    UnitState,
)
from pyrpoc.plugins.program_io.requests import (
    UNIT_COMMAND,
    AddUnit,
    EditUnit,
    PauseUnit,
    RemoveUnit,
    ResumeUnit,
    RunUnitNow,
    UnitCommand,
)
from pyrpoc.utils.widgets.data_panel import DataPanel, data_panel_registry
from pyrpoc.utils.widgets.source_picker import outputs_of_type

from .activity import Activity
from .canvas import UnitCanvas
from .guidance import LoopStage, PanelState, duration, guidance, loop_stage
from .loop_watch import LoopWatch
from .picture import backdrop
from .shown import ShownUnit, drawn, in_loop
from .units_table import UnitsTable

# A loop logs many events a second; the panel redraws at most this often.
REDRAW_MS = 250


def loop_of(library: Library, survey: SurveyOutput) -> LoopLogOutput | None:
    """The log of the newest open loop that read ``survey``."""
    location = survey.recording.location
    if location is None:
        return None
    return next(
        (
            log
            for log in outputs_of_type(library, LoopLogOutput)
            if log.recording.folder == location.parent
            and any(read.recording == location.name for read in log.recording.reads.values())
        ),
        None,
    )


@data_panel_registry.register("autonomy")
class AutonomyPanel(DataPanel[SurveyOutput]):
    display_name = "Sites"
    group = "Multisite"
    order = 40
    renders = SurveyOutput

    # The library or the followed loop changed, on whatever thread changed it.
    outside_changed = pyqtSignal()

    def __init__(self, library: Library, inputs: ProgramInputs, requests: Requests):
        super().__init__(library, inputs, requests)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.chosen: str | None = None
        # What is drawn, so a redraw with nothing new does no work.
        self.backdrop_of: SurveyOutput | None = None
        self.loop: LoopLogOutput | None = None
        self.loop_view: LoopView | None = None
        self.activity = Activity()

        root = QVBoxLayout(self.body)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self.build_tools())
        root.addWidget(self.build_splitter(), 1)
        self.status = QLabel("", self.body)
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.redraw_soon = QTimer(self)
        self.redraw_soon.setSingleShot(True)
        self.redraw_soon.setInterval(REDRAW_MS)
        self.redraw_soon.timeout.connect(self.follow_loop)
        self.connect_signals()
        self.connect_source()

    def build_tools(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.draw_button = QPushButton("Draw site", self.body)
        self.draw_button.setCheckable(True)
        self.draw_button.setToolTip(
            "Click each corner; double-click, or click the first corner, to close it"
        )
        row.addWidget(self.draw_button)
        self.delete_button = QPushButton("Remove", self.body)
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
        self.loop_watch.visit_next.connect(self.ask_visit_next)
        self.loop_watch.pause.connect(self.toggle_pause)
        self.outside_changed.connect(self.redraw_soon.start)
        self.inputs.subscribe(self.on_input_changed)
        self.library.subscribe(self.on_collection_changed)
        # A run ending changes only what is accepted, so this is where the panel learns that its
        # loop stopped.
        self.requests.subscribe(self.on_collection_changed)

    @property
    def running(self) -> bool:
        """Whether a loop is running on this survey, so changes go to it as commands."""
        return self.loop is not None and not self.loop.recording.finished

    @property
    def editable(self) -> bool:
        """Units can be changed before any loop, or while one runs; once one has stopped, they
        are its record."""
        return self.loop is None or self.running

    def units(self) -> tuple[ShownUnit, ...]:
        if self.loop_view is not None:
            return in_loop(self.loop_view.units)
        return drawn(self.inputs.get(UNITS.key))

    def draw(self, program_output: SurveyOutput) -> None:
        self.show_backdrop(program_output)
        self.follow_loop()

    def show_backdrop(self, survey: SurveyOutput) -> None:
        """The survey's overview, once it has stitched one."""
        if self.backdrop_of is survey or not survey.has_overview():
            return
        self.backdrop_of = survey
        rgb, rect = backdrop(*survey.overview())
        self.canvas.show_backdrop(rgb, rect)

    def on_input_changed(self, key: str) -> None:
        if key == UNITS.key:
            self.show_units()

    def follow_loop(self) -> None:
        """Follow the loop on the shown survey, if there is one, and show what it is doing."""
        survey = self.shown_output
        loop = None if survey is None else loop_of(self.library, survey)
        if loop is not self.loop:
            if self.loop is not None:
                self.loop.unsubscribe(self.on_loop_changed)
            if loop is not None:
                loop.subscribe(self.on_loop_changed)
            self.loop, self.activity = loop, Activity()
        if survey is not None:
            self.show_backdrop(survey)
        self.loop_view = None if loop is None else loop.view()
        self.loop_watch.setVisible(loop is not None)
        if loop is not None:
            self.show_loop(loop)
        self.show_units()

    def show_loop(self, loop: LoopLogOutput) -> None:
        """How long the loop has run, and what it did since last shown."""
        new = loop.events_after(self.activity.taken)
        self.activity.take(new)
        if new or self.loop_watch.activity.count() == 0:
            self.loop_watch.show_activity(list(self.activity.lines))
        heading = {
            LoopStage.RUNNING: "Loop running, {} so far",
            LoopStage.STOPPED_EARLY: "Loop stopped early, {} in",
        }.get(self.stage(), "Loop finished, {} in all")
        self.loop_watch.show_heading(heading.format(duration(self.minutes())))

    def stage(self) -> LoopStage:
        return loop_stage(self.loop_view, self.running)

    def minutes(self) -> float:
        return 0.0 if self.loop_view is None else self.loop_view.elapsed_s / 60.0

    def on_loop_changed(self, _output: ProgramOutput) -> None:
        # On the run's thread: hand over to the GUI thread.
        self.outside_changed.emit()

    def on_collection_changed(self) -> None:
        # A part begins on the run's thread, so this too may not be the GUI thread.
        self.outside_changed.emit()

    def show_units(self) -> None:
        units = self.units()
        chosen = next((unit for unit in units if unit.key == self.chosen), None)
        self.chosen = None if chosen is None else chosen.key
        # Next visits mean nothing once the loop is not running.
        elapsed = None if self.loop_view is None or not self.running else self.loop_view.elapsed_s
        self.canvas.show_units(units, self.chosen, self.editable)
        self.table.show_units(units, self.chosen, self.editable, elapsed)
        self.draw_button.setEnabled(self.editable)
        if not self.editable:
            self.draw_button.setChecked(False)
        self.delete_button.setEnabled(self.editable and chosen is not None)
        paused = chosen is not None and chosen.state is UnitState.PAUSED
        self.loop_watch.set_site(None if chosen is None else chosen.number, paused, self.running)
        self.status.setText(guidance(self.panel_state(len(units))))

    def panel_state(self, sites: int) -> PanelState:
        survey = self.shown_output
        return PanelState(
            has_survey=survey is not None,
            has_map=survey is not None and survey.has_overview(),
            drawing=self.draw_button.isChecked(),
            sites=sites,
            stage=self.stage(),
            minutes=self.minutes(),
        )

    def choose(self, key: str | None) -> None:
        if key != self.chosen:
            self.chosen = key
            self.show_units()

    def chosen_unit(self) -> ShownUnit | None:
        return next((unit for unit in self.units() if unit.key == self.chosen), None)

    def command(self, command: UnitCommand) -> None:
        """Send ``command`` to the running loop."""
        if not self.requests.post(UNIT_COMMAND, command):
            self.status.setText("No loop is running, so nothing was changed")

    def ask_visit_next(self) -> None:
        if self.chosen is not None:
            self.command(RunUnitNow(self.chosen))

    def toggle_pause(self) -> None:
        unit = self.chosen_unit()
        if unit is None:
            return
        paused = unit.state is UnitState.PAUSED
        self.command(ResumeUnit(unit.key) if paused else PauseUnit(unit.key))

    def on_draw_toggled(self, on: bool) -> None:
        if on:
            self.canvas.start_drawing()
        else:
            self.canvas.stop_drawing()
        self.show_units()

    def on_drawn(self, outline: Outline) -> None:
        unit = Unit.new(outline, "")
        self.chosen = unit.key
        if self.running:
            self.command(AddUnit(unit))
        else:
            self.inputs.set(UNITS.key, (*self.inputs.get(UNITS.key), unit))

    def on_reshaped(self, key: str, outline: Outline) -> None:
        self.change(key, lambda unit: replace(unit, outline=outline))

    def on_regrouped(self, key: str, group: str) -> None:
        self.change(key, lambda unit: replace(unit, group=group))

    def change(self, key: str, edit: Callable[[Unit], Unit]) -> None:
        """Give the unit with ``key`` its edited outline or group."""
        if self.running:
            unit = next(each for each in self.units() if each.key == key).unit
            self.command(EditUnit(edit(unit)))
            return
        units: tuple[Unit, ...] = self.inputs.get(UNITS.key)
        self.inputs.set(UNITS.key, tuple(edit(u) if u.key == key else u for u in units))

    def delete_chosen(self) -> None:
        chosen, self.chosen = self.chosen, None
        if chosen is None:
            return
        if self.running:
            self.command(RemoveUnit(chosen))
            return
        units: tuple[Unit, ...] = self.inputs.get(UNITS.key)
        self.inputs.set(UNITS.key, tuple(unit for unit in units if unit.key != chosen))

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
        self.follow_loop()

    def detach(self) -> None:
        self.redraw_soon.stop()
        if self.loop is not None:
            self.loop.unsubscribe(self.on_loop_changed)
        self.inputs.unsubscribe(self.on_input_changed)
        self.library.unsubscribe(self.on_collection_changed)
        self.requests.unsubscribe(self.on_collection_changed)
        super().detach()
