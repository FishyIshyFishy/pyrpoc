"""Units drawn over a survey's overview: draw, move, reshape, group and delete them. They are
the ``UNITS`` program input, and are kept in ``units.json`` in the survey's folder."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout

from pyrpoc.core.library import Library
from pyrpoc.core.panels import DataPanel, data_panel_registry
from pyrpoc.core.program_input import ParameterError, ProgramInputs
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.plugins.program_io.inputs import UNITS, Unit
from pyrpoc.plugins.program_io.outputs import Mosaic, SurveyResult

from .canvas import UnitCanvas
from .picture import backdrop, tinted
from .units_file import read_units, write_units
from .units_table import UnitsTable


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

        root = QVBoxLayout(self.body)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self.build_tools())
        splitter = QSplitter(Qt.Orientation.Horizontal, self.body)
        splitter.setChildrenCollapsible(False)
        self.canvas = UnitCanvas(splitter)
        self.table = UnitsTable(splitter)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([700, 260])
        root.addWidget(splitter, 1)
        self.status = QLabel("", self.body)
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.canvas.drawn.connect(self.on_drawn)
        self.canvas.reshaped.connect(self.on_reshaped)
        self.canvas.selected.connect(self.choose)
        self.table.chosen.connect(self.choose)
        self.table.regrouped.connect(self.on_regrouped)
        inputs.subscribe(self.on_input_changed)
        self.connect_source()
        self.show_units()

    def build_tools(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.draw_button = QPushButton("Draw unit", self.body)
        self.draw_button.setCheckable(True)
        self.draw_button.setToolTip(
            "Click each corner; double-click, or click the first corner, to close it"
        )
        self.draw_button.toggled.connect(self.on_draw_toggled)
        row.addWidget(self.draw_button)
        self.delete_button = QPushButton("Delete", self.body)
        self.delete_button.clicked.connect(self.delete_chosen)
        row.addWidget(self.delete_button)
        row.addStretch(1)
        return row

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
        if key == UNITS.key:
            self.show_units()

    def show_units(self) -> None:
        units = self.units
        if self.chosen not in {unit.id for unit in units}:
            self.chosen = None
        self.canvas.show_units(units, self.chosen)
        self.table.show_units(units, self.chosen)
        self.delete_button.setEnabled(self.chosen is not None)
        if self.folder is not None:
            self.status.setText(f"{len(units)} units, kept in this survey's folder")

    def choose(self, unit_id: int | None) -> None:
        if unit_id != self.chosen:
            self.chosen = unit_id
            self.show_units()

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
        self.inputs.unsubscribe(self.on_input_changed)
        super().detach()
