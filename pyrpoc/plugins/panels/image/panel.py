"""The image panel: one tile per channel, each with its own autoscale and LUT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core.data import Image
from pyrpoc.core.library import Library
from pyrpoc.core.panels import DataPanel, data_panel_registry
from pyrpoc.core.pixel_pick import PICK, PixelPick
from pyrpoc.core.program_inputs import ProgramInputs
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.shared.widgets.levels import autoscale_levels, mono_levels

LUT = pg.ColorMap(
    pos=np.array([0.0, 0.999, 1.0], dtype=float),
    color=np.array([[0, 0, 0, 255], [255, 255, 255, 255], [255, 0, 0, 255]], dtype=np.ubyte),
)


@dataclass
class ChannelTile:
    root: QWidget
    autoscale_box: QCheckBox
    plot: pg.PlotWidget
    image_item: pg.ImageItem
    hist_widget: pg.HistogramLUTWidget
    min_val: float = 0.0
    max_val: float = 1.0


@dataclass
class TileState:
    """One tile's saved settings, parsed from the session."""

    index: int
    autoscale: bool
    min_val: float
    max_val: float


@data_panel_registry.register("image")
class ImagePanel(DataPanel[Image]):
    display_name = "Tiled channels"
    group = "2D Image Views"
    order = 21
    renders = Image

    def __init__(self, library: Library, inputs: ProgramInputs):
        super().__init__(library, inputs)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._tiles: list[ChannelTile] = []
        self._pending_state: list[TileState] = []
        self._suspend_lut_signal = False

        outer = QVBoxLayout(self.body)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self.body)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        self._content = QWidget(scroll)
        self._grid = QGridLayout(self._content)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(10)
        scroll.setWidget(self._content)
        self.connect_source()

    def show_value(self, program_output: ProgramOutput[Image], value: Image) -> None:
        del program_output
        self.sync_channel_tiles(len(value.channels))
        for index, plane in enumerate(value.array):
            self.update_channel_image(index, plane)

    def clear(self) -> None:
        self.sync_channel_tiles(0)

    def sync_channel_tiles(self, count: int) -> None:
        """Add or drop tiles to match ``count``, leaving the levels and zoom of tiles that
        stay."""
        if len(self._tiles) == count:
            return
        while len(self._tiles) > count:
            tile = self._tiles.pop()
            # Hidden, not unparented: a show the layout queued when it was added
            # would otherwise open it as a top-level window before deletion.
            tile.root.hide()
            tile.root.deleteLater()
        while len(self._tiles) < count:
            self._tiles.append(self.build_tile(len(self._tiles)))
        self.apply_pending_state()
        self.reflow_tiles()

    def build_tile(self, index: int) -> ChannelTile:
        root = QWidget(self._content)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        body = QHBoxLayout()
        root_layout.addLayout(body, 1)

        plot, image_item = self.build_plot(root)
        body.addWidget(plot, 1)
        hist_widget = pg.HistogramLUTWidget(root)
        hist_widget.setImageItem(image_item)
        hist_widget.item.gradient.setColorMap(LUT)
        autoscale_box = QCheckBox("Autoscale", root)
        autoscale_box.setChecked(True)
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(4)
        right_col.addWidget(hist_widget, 1)
        right_col.addWidget(autoscale_box)
        body.addLayout(right_col)

        tile = ChannelTile(root, autoscale_box, plot, image_item, hist_widget)
        autoscale_box.toggled.connect(lambda _checked, i=index: self.on_autoscale_toggled(i))
        hist_widget.item.sigLevelsChanged.connect(
            lambda _item, i=index: self.on_lut_levels_changed(i)
        )
        return tile

    def build_plot(self, parent: QWidget) -> tuple[pg.PlotWidget, pg.ImageItem]:
        """A tile's plot and image, wired to pick the pixel under a click."""
        plot = pg.PlotWidget(parent)
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setAspectLocked(True)
        plot.invertY(True)
        image_item = pg.ImageItem()
        image_item.setColorMap(LUT)
        plot.addItem(image_item)
        scene = plot.sceneObj
        if scene is None:
            raise RuntimeError("a new PlotWidget has no scene")
        scene.sigMouseClicked.connect(lambda event: self.on_scene_clicked(event, image_item))
        return plot, image_item

    def on_scene_clicked(self, event: Any, image_item: pg.ImageItem) -> None:
        """Pick the pixel under a left click."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = image_item.mapFromScene(event.scenePos())
        if self.pick_pixel(int(np.floor(point.x())), int(np.floor(point.y()))):
            event.accept()

    def pick_pixel(self, x: int, y: int) -> bool:
        """Set the pick to pixel ``(x, y)`` of the image on screen, returning whether it is a
        pixel of that image."""
        # Checked against the frame: a click just outside the pixels is still
        # inside the view box, and an out-of-range pixel would park the galvos
        # outside the scan.
        program_output, image = self.shown_output, self.shown_value
        if program_output is None or image is None:
            return False
        _, height, width = image.array.shape
        if not (0 <= x < width and 0 <= y < height):
            return False
        self.inputs.set(PICK.key, PixelPick.on(program_output, x, y))
        return True

    def on_autoscale_toggled(self, index: int) -> None:
        image = self.shown_value
        if image is not None:
            self.update_channel_image(index, image.array[index])

    def on_lut_levels_changed(self, index: int) -> None:
        if self._suspend_lut_signal:
            return
        tile = self._tiles[index]
        min_val, max_val = mono_levels(tile.hist_widget)
        if max_val <= min_val:
            max_val = min_val + 1e-12
            tile.hist_widget.item.setLevels(min_val, max_val)
        tile.min_val = min_val
        tile.max_val = max_val

    def update_channel_image(self, index: int, channel: np.ndarray) -> None:
        tile = self._tiles[index]
        tile.image_item.setImage(channel, autoLevels=False)
        if tile.autoscale_box.isChecked():
            self.apply_levels(tile, *autoscale_levels(channel))
        else:
            self.apply_levels(tile, *mono_levels(tile.hist_widget))

    def apply_levels(self, tile: ChannelTile, min_val: float, max_val: float) -> None:
        tile.min_val = min_val
        tile.max_val = max_val
        tile.image_item.setLevels((min_val, max_val))
        self._suspend_lut_signal = True
        try:
            tile.hist_widget.item.setLevels(min_val, max_val)
        finally:
            self._suspend_lut_signal = False

    def reflow_tiles(self) -> None:
        while (item := self._grid.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.setParent(self._content)

        columns = min(2, len(self._tiles))
        rows = (len(self._tiles) + columns - 1) // columns if columns else 0
        for index, tile in enumerate(self._tiles):
            self._grid.addWidget(tile.root, index // columns, index % columns)
        # The grid keeps the row and column count it once had; stretch left on
        # an emptied one would reserve blank space beside the remaining tiles.
        for row in range(self._grid.rowCount()):
            self._grid.setRowStretch(row, 1 if row < rows else 0)
        for col in range(self._grid.columnCount()):
            self._grid.setColumnStretch(col, 1 if col < columns else 0)

    def export_persistence_state(self) -> dict[str, Any]:
        return {
            "channels": [
                {
                    "index": index,
                    "autoscale": tile.autoscale_box.isChecked(),
                    "min_val": tile.min_val,
                    "max_val": tile.max_val,
                }
                for index, tile in enumerate(self._tiles)
            ],
        }

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        """Take saved tile settings from session JSON, skipping malformed rows."""
        # Applied once tiles exist, since none do yet.
        channels = state.get("channels", [])
        if not isinstance(channels, list):
            return
        self._pending_state = [
            TileState(
                index=int(row.get("index", position)),
                autoscale=bool(row.get("autoscale", True)),
                min_val=float(row.get("min_val", 0.0)),
                max_val=float(row.get("max_val", 1.0)),
            )
            for position, row in enumerate(channels)
            if isinstance(row, dict)
        ]
        self.apply_pending_state()

    def apply_pending_state(self) -> None:
        """Apply saved tile settings, once."""
        # Once, so live edits during a run are not overwritten on every frame.
        if not self._pending_state or not self._tiles:
            return
        for saved in self._pending_state:
            if not 0 <= saved.index < len(self._tiles):
                continue
            tile = self._tiles[saved.index]
            tile.autoscale_box.blockSignals(True)
            tile.autoscale_box.setChecked(saved.autoscale)
            tile.autoscale_box.blockSignals(False)
            self.apply_levels(tile, saved.min_val, max(saved.max_val, saved.min_val + 1e-12))
        self._pending_state = []
