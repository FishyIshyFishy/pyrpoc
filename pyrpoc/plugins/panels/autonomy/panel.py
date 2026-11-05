"""Units drawn over a survey's overview: draw, move, reshape, group and delete them, and while
a loop runs on them, watch and steer it. Units are the ``UNITS`` program input, kept in
``units.json`` in the survey's folder."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget

from pyrpoc.core.library import Library
from pyrpoc.core.panels import DataPanel, data_panel_registry
from pyrpoc.core.program_input import ParameterError, ProgramInputs
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.plugins.program_io.inputs import PAUSED_UNITS, RUN_NOW, UNITS, RunNow, Unit
from pyrpoc.plugins.program_io.outputs import LoopState, Mosaic, SurveyResult, UnitState

from .canvas import UnitCanvas
from .loop_view import LoopView
from .picture import backdrop, tinted
from .units_file import read_units, write_units
from .units_table import UnitsTable

# How often a running loop's state is looked at again.
WATCH_MS = 500


def same_run(a: ProgramOutput[Any], b: ProgramOutput[Any]) -> bool:
    return a.provenance.started_at == b.provenance.started_at and a.run_id == b.run_id


@data_panel_registry.register("autonomy")
class AutonomyPanel(DataPanel[SurveyResult]):
    display_name = "Autonomy"
    group = "Autonomous"
    order = 40
    renders = SurveyResult

    def __init__(self, library: Library, inputs: ProgramInputs):
        super().__init__(library, inputs)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.folder: Path | None = None
        self.chosen: int | None = None
        self.backdrop_of: str | None = None
        self.loop: ProgramOutput[LoopState] | None = None

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
        self.loop_view = LoopView(side)
        self.loop_view.setVisible(False)
        layout.addWidget(self.loop_view, 1)
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
        self.loop_view.run_now.connect(self.ask_run_now)
        self.loop_view.pause.connect(self.toggle_pause)
        self.inputs.subscribe(self.on_input_changed)

    @property
    def units(self) -> tuple[Unit, ...]:
        return self.inputs.get(UNITS.key)

    def show_value(self, program_output: ProgramOutput[SurveyResult], value: SurveyResult) -> None:
        del value
        self.show_backdrop(program_output)
        meta_path = program_output.meta_path
        folder = None if meta_path is None else meta_path.parent
        if folder != self.folder:
            self.folder = folder
            self.load_units()

    def show_backdrop(self, survey: ProgramOutput[SurveyResult]) -> None:
        """The overview of the survey's own mosaic, once it is in the library."""
        mosaics = [each for each in self.library.matching(Mosaic) if same_run(each, survey)]
        mosaic = mosaics[0].value if mosaics else None
        if mosaic is None or not mosaic.tiles or self.backdrop_of == mosaics[0].id:
            return
        self.backdrop_of = mosaics[0].id
        picture = backdrop(mosaic)
        rect = QRectF(picture.left_um, picture.top_um, picture.width_um, picture.height_um)
        self.canvas.show_backdrop(tinted(picture.planes), rect)

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

    def loop_state(self) -> LoopState | None:
        return None if self.loop is None else self.loop.value

    def show_units(self) -> None:
        units = self.units
        if self.chosen not in {unit.id for unit in units}:
            self.chosen = None
        state = self.loop_state()
        states: dict[int, UnitState] = {} if state is None else {u.id: u for u in state.units}
        elapsed = 0.0 if state is None else state.elapsed_s
        self.canvas.show_units(units, self.chosen)
        self.table.show_units(units, self.chosen, states, elapsed)
        self.delete_button.setEnabled(self.chosen is not None)
        running = self.loop is not None and not self.loop.finished
        paused = self.chosen in self.inputs.get(PAUSED_UNITS.key)
        self.loop_view.set_unit(self.chosen is not None, paused, running)
        if self.folder is not None:
            self.status.setText(f"{len(units)} units, kept in this survey's folder")

    def follow_loop(self) -> None:
        """Show the newest loop's state, while there is one."""
        loops = self.library.matching(LoopState)
        self.loop = max(loops, key=lambda each: each.provenance.started_at) if loops else None
        state = self.loop_state()
        self.loop_view.setVisible(state is not None)
        if self.loop is None or state is None:
            self.canvas.show_tracked([])
            return
        running = not self.loop.finished
        self.loop_view.show_state(state, running)
        self.canvas.show_tracked([unit.outline for unit in state.units if unit.state != "done"])
        self.show_units()

    def choose(self, unit_id: int | None) -> None:
        if unit_id != self.chosen:
            self.chosen = unit_id
            self.show_units()

    def ask_run_now(self) -> None:
        if self.chosen is None:
            return
        asked: tuple[RunNow, ...] = self.inputs.get(RUN_NOW.key)
        serial = max((request.serial for request in asked), default=0) + 1
        self.inputs.set(RUN_NOW.key, (*asked, RunNow(serial, self.chosen)))

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
        super().detach()
