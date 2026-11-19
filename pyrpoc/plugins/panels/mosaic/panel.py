"""The Mosaic panel: a mosaic laid on the dish, each tile where the stage put it and then the
stitched overview, with every channel overlaid in its own colour. Regions marked on it are kept
here to come back to: right-click one to send the stage there, or to delete it."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QLabel, QMenu, QPushButton, QSplitter, QVBoxLayout

from pyrpoc.core.library import Library
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.plugins.program_io.outputs import MosaicOutput
from pyrpoc.utils.widgets.data_panel import DataPanel, data_panel_registry
from pyrpoc.utils.widgets.stage_link import StageLink

from .canvas import MosaicCanvas
from .channels import ChannelStrip
from .regions import Regions

# How long a note, such as where the stage was sent, stays in place of the status line, in s.
NOTICE_S = 6.0


def action(menu: QMenu, text: str, enabled: bool, does: Callable[[], None]) -> None:
    item = QAction(text, menu)
    item.setEnabled(enabled)
    item.triggered.connect(lambda _checked=False: does())
    menu.addAction(item)


@data_panel_registry.register("mosaic")
class MosaicPanel(DataPanel[MosaicOutput]):
    display_name = "Mosaic"
    group = "Images"
    order = 22
    renders = MosaicOutput

    def __init__(self, library: Library, inputs: ProgramInputs, stage: StageLink):
        super().__init__(library, inputs, stage)
        self.regions = Regions()
        self.notice: tuple[str, float] | None = None
        self.build_buttons()
        root = QVBoxLayout(self.body)
        root.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel("", self.body)
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        splitter = QSplitter(Qt.Orientation.Horizontal, self.body)
        splitter.setChildrenCollapsible(False)
        self.canvas = MosaicCanvas(splitter, self.regions)
        self.channels = ChannelStrip(splitter)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.channels)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([800, 300])
        root.addWidget(splitter, 1)
        self.connect_canvas()
        # The note in place of the status line expires on its own.
        self.clock = QTimer(self)
        self.clock.setInterval(1000)
        self.clock.timeout.connect(self.show_status)
        self.clock.start()
        self.connect_source()

    def build_buttons(self) -> None:
        row = self.source.row
        self.draw_button = QPushButton("Mark region", self)
        self.draw_button.setCheckable(True)
        self.draw_button.setToolTip(
            "Drag on the mosaic round what to come back to; the region kept is the dotted square "
            "one field across. Esc stops"
        )
        self.fit_button = QPushButton("Whole mosaic", self)
        for button in (self.draw_button, self.fit_button):
            row.addWidget(button)

    def connect_canvas(self) -> None:
        canvas = self.canvas
        self.draw_button.toggled.connect(canvas.set_drawing)
        self.fit_button.clicked.connect(canvas.fit_all)
        canvas.drawing_changed.connect(self.draw_button.setChecked)
        canvas.chose.connect(canvas.select)
        canvas.drawn.connect(self.add_region)
        canvas.refused.connect(self.note)
        canvas.menu_asked.connect(self.show_menu)
        self.channels.changed.connect(self.recolour)

    def draw(self, program_output: MosaicOutput) -> None:
        if not len(program_output):
            self.show_status()
            return
        if program_output.is_stitched:
            planes, _placement = program_output.overview()
        else:
            planes = program_output[len(program_output) - 1]
        # Levels are judged on the overview once there is one, else on the newest tile.
        self.channels.set_planes(list(planes), list(program_output.channels))
        self.canvas.show_mosaic(program_output, self.channels.colour)
        self.canvas.recolour(self.channels.colour)
        self.show_status()

    def recolour(self) -> None:
        if self.channels.planes:
            self.canvas.recolour(self.channels.colour)

    def clear(self) -> None:
        self.channels.clear_planes()
        self.canvas.clear_mosaic()
        self.show_status()

    def show_status(self) -> None:
        if self.notice is not None and time.monotonic() - self.notice[1] < NOTICE_S:
            self.status.setText(self.notice[0])
            return
        self.status.setText(self.progress())

    def progress(self) -> str:
        mosaic = self.shown_output
        if mosaic is None or not mosaic.tiles:
            return ""
        taken = f"{len(mosaic)}/{len(mosaic.tiles)} tiles"
        if mosaic.is_stitched:
            return f"{taken}, stitched · {mosaic.field_um:.0f} µm fields"
        if mosaic.recording.finished:
            return f"{taken}, cut short: shown where the stage put them"
        return f"{taken}, shown where the stage put them until stitched"

    def note(self, text: str) -> None:
        self.notice = (text, time.monotonic())
        self.show_status()

    def add_region(self, x_um: float, y_um: float) -> None:
        if self.canvas.field_um is None:
            return
        region = self.regions.add(x_um, y_um, self.canvas.field_um)
        self.canvas.select(region.number)

    def delete_region(self, number: int) -> None:
        self.regions.remove(number)
        self.canvas.select(None)

    def move_stage_to(self, number: int) -> None:
        region = self.regions.get(number)
        refusal = self.stage.move_to(region.x_um, region.y_um)
        if refusal is None:
            self.note(f"Stage moving to region {number} ({region.x_um:.1f}, {region.y_um:.1f} µm)")
        else:
            self.note(f"Stage not moved: {refusal}")

    def show_menu(self, number: int | None, at: QPoint) -> None:
        menu = QMenu(self)
        if number is None:
            action(menu, "Mark a region", self.canvas.field_um is not None, self.start_marking)
            action(menu, "Show whole mosaic", True, self.canvas.fit_all)
        else:
            self.region_menu(menu, number)
        menu.exec(at)

    def region_menu(self, menu: QMenu, number: int) -> None:
        region = self.regions.get(number)
        # A title line, since not every style draws a menu's section titles.
        action(
            menu, f"Region {number} ({region.x_um:.1f}, {region.y_um:.1f} µm)", False, lambda: None
        )
        menu.addSeparator()
        blocked = self.stage.blocked()
        move = "Move stage here" if blocked is None else f"Move stage here ({blocked})"
        action(menu, move, blocked is None, lambda: self.move_stage_to(number))
        action(menu, "Delete region", True, lambda: self.delete_region(number))

    def start_marking(self) -> None:
        self.draw_button.setChecked(True)

    def export_persistence_state(self) -> dict[str, Any]:
        return {"regions": self.regions.export_state(), "channels": self.channels.export_state()}

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        """Take saved settings from session JSON; a malformed entry keeps its default."""
        self.regions.import_state(state.get("regions"))
        self.canvas.show_regions()
        self.channels.import_state(state.get("channels"))
        self.refresh()
