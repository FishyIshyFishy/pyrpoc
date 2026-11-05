"""Any mosaic: its tiles placed on the dish as they arrive, then its stitched overview, with
every channel overlaid in its own colour and the axes in dish µm."""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout

from pyrpoc.core.library import Library
from pyrpoc.core.panels import DataPanel, data_panel_registry
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.plugins.program_io.outputs import Mosaic
from pyrpoc.plugins.program_io.outputs.mosaic_view import Picture, overview_picture, placed_tiles
from pyrpoc.utils.widgets.channel_strip import ChannelStrip
from pyrpoc.utils.widgets.colour_image import ColourImage

# The longest side the picture is drawn at; a large survey is shrunk to this.
DISPLAY_PX = 2048


@data_panel_registry.register("mosaic")
class MosaicPanel(DataPanel[Mosaic]):
    display_name = "Mosaic"
    group = "2D Image Views"
    order = 22
    renders = Mosaic

    def __init__(self, library: Library, inputs: ProgramInputs):
        super().__init__(library, inputs)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.picture: Picture | None = None
        # What the picture was made from, so a repeat notification draws nothing new.
        self.drawn: tuple[str, int, bool, bool] | None = None

        root = QVBoxLayout(self.body)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal, self.body)
        splitter.setChildrenCollapsible(False)
        self.image = ColourImage(splitter)
        self.channels = ChannelStrip(splitter)
        splitter.addWidget(self.image)
        splitter.addWidget(self.channels)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([800, 300])
        root.addWidget(splitter, 1)
        root.addLayout(self.build_footer())

        self.channels.changed.connect(self.recolour)
        scene = self.image.sceneObj
        if scene is None:
            raise RuntimeError("a new PlotWidget has no scene")
        scene.sigMouseMoved.connect(self.on_mouse_moved)
        self.connect_source()

    def build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.status = QLabel("", self.body)
        row.addWidget(self.status, 1)
        self.position = QLabel("", self.body)
        row.addWidget(self.position)
        # Rarely wanted, so it is only offered once there is an overview to switch from.
        self.raw_tiles = QCheckBox("Raw tiles", self.body)
        self.raw_tiles.setToolTip("Show the tiles where the stage put them, not stitched")
        self.raw_tiles.setVisible(False)
        self.raw_tiles.toggled.connect(lambda _on: self.redraw())
        row.addWidget(self.raw_tiles)
        return row

    def show_value(self, program_output: ProgramOutput[Mosaic], value: Mosaic) -> None:
        if not value.tiles:
            self.clear()
            return
        stitched = value.overview is not None
        raw = self.raw_tiles.isChecked() or not stitched
        key = (program_output.id, len(value.tiles), stitched, raw)
        self.raw_tiles.setVisible(stitched)
        self.status.setText(value.summary())
        if key == self.drawn:
            return
        self.drawn = key
        self.picture = (
            placed_tiles(value, DISPLAY_PX) if raw else overview_picture(value, DISPLAY_PX)
        )
        self.channels.set_planes(list(self.picture.planes), list(value.channels))
        self.recolour()

    def redraw(self) -> None:
        """Draw the shown mosaic again, as the raw-tiles choice now says."""
        if self.shown_output is not None and self.shown_value is not None:
            self.show_value(self.shown_output, self.shown_value)

    def recolour(self) -> None:
        picture = self.picture
        if picture is None or not self.channels.planes:
            return
        rect = QRectF(picture.left_um, picture.top_um, picture.width_um, picture.height_um)
        self.image.show_rgb(self.channels.rgb(), rect)

    def on_mouse_moved(self, scene_position: QPointF) -> None:
        """Name the dish position under the mouse."""
        picture = self.picture
        if picture is None:
            return
        pixel = self.image.item.mapFromScene(scene_position)
        _, height, width = picture.planes.shape
        x = picture.left_um + pixel.x() * picture.width_um / width
        y = picture.top_um + pixel.y() * picture.height_um / height
        self.position.setText(f"x {x:.1f} µm · y {y:.1f} µm")

    def clear(self) -> None:
        self.picture, self.drawn = None, None
        self.channels.clear_planes()
        self.image.clear_image()
        self.status.setText("")
        self.position.setText("")

    def export_persistence_state(self) -> dict[str, Any]:
        return {"channels": self.channels.export_state()}

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        """Take saved settings from session JSON; a malformed entry keeps its default."""
        self.channels.import_state(state.get("channels"))
        self.redraw()
