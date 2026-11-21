"""The image panel: one tile per channel, each with its own autoscale and LUT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core.library import Library
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.plugins.program_io.inputs import PICK, PixelPick
from pyrpoc.plugins.program_io.outputs import ImageOutput
from pyrpoc.utils.widgets.data_panel import DataPanel, data_panel_registry
from pyrpoc.utils.widgets.frame_slider import FrameSlider
from pyrpoc.utils.widgets.icons import asset_icon
from pyrpoc.utils.widgets.levels import (
    LevelsControl,
    LevelsState,
    autoscale_levels,
    parse_levels,
)
from pyrpoc.utils.widgets.stage_link import StageLink

from .measure import Shape, TileMeasurement, Tool
from .measure_settings import MeasureSettings

LUT = pg.ColorMap(
    pos=np.array([0.0, 0.999, 1.0], dtype=float),
    color=np.array([[0, 0, 0, 255], [255, 255, 255, 255], [255, 0, 0, 255]], dtype=np.ubyte),
)


@dataclass
class ChannelTile:
    root: QWidget
    plot: pg.PlotWidget
    levels: LevelsControl
    measure: TileMeasurement


@data_panel_registry.register("image")
class ImagePanel(DataPanel[ImageOutput]):
    display_name = "Tiled"
    group = "Images"
    order = 21
    renders = ImageOutput

    def __init__(self, library: Library, inputs: ProgramInputs, stage: StageLink):
        super().__init__(library, inputs, stage)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._tiles: list[ChannelTile] = []
        self._pending_state: list[LevelsState] = []
        self.tool: Tool | None = None
        # The ``(C, H, W)`` image on screen.
        self.shown_image: np.ndarray | None = None
        self.shape: Shape | None = None
        self.measure_settings = MeasureSettings()
        self.tool_buttons = self.build_tool_buttons()

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
        self.frames = FrameSlider(self.body)
        self.frames.moved.connect(self.refresh)
        outer.addWidget(self.frames)
        self.connect_source()

    def build_tool_buttons(self) -> dict[Tool, QToolButton]:
        """Two small toggles beside the source picker. With neither on, a click picks a pixel."""
        tips = {
            Tool.LINE: "Line profile: drag across an image to plot the values along the line",
            Tool.REGION: "Region histogram: drag a box to plot the values inside it",
        }
        icons = {Tool.LINE: "profile", Tool.REGION: "region"}
        buttons: dict[Tool, QToolButton] = {}
        for tool in Tool:
            button = QToolButton(self)
            button.setCheckable(True)
            button.setAutoRaise(True)
            button.setIcon(asset_icon(icons[tool]))
            button.setToolTip(tips[tool])
            button.toggled.connect(lambda checked, t=tool: self.on_tool_toggled(t, checked))
            self.source.row.addWidget(button)
            buttons[tool] = button
        return buttons

    def on_tool_toggled(self, tool: Tool, checked: bool) -> None:
        if checked:
            for other, button in self.tool_buttons.items():
                if other is not tool and button.isChecked():
                    button.blockSignals(True)
                    button.setChecked(False)
                    button.blockSignals(False)
            self.set_tool(tool)
        elif self.tool is tool:
            self.set_tool(None)

    def set_tool(self, tool: Tool | None) -> None:
        self.tool = tool
        for each in self._tiles:
            each.measure.show_hint_for(tool)
            cursor = Qt.CursorShape.CrossCursor if tool is not None else Qt.CursorShape.ArrowCursor
            each.plot.setCursor(cursor)

    def on_shape_changed(self, source: TileMeasurement, shape: Shape | None) -> None:
        """Draw a shape one tile changed on all the others."""
        self.shape = shape
        for tile in self._tiles:
            if tile.measure is not source:
                tile.measure.apply_shape(shape)

    def draw(self, program_output: ImageOutput) -> None:
        """Draw the image the frame slider is on, which follows the newest while on the last."""
        if not len(program_output):
            return
        self.frames.take_count(len(program_output))
        self.shown_image = program_output[self.frames.index()]
        self.sync_channel_tiles(len(program_output.channels))
        for index, plane in enumerate(self.shown_image):
            self.update_channel_image(index, plane)

    def clear(self) -> None:
        self.shown_image = None
        self.frames.reset()
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
            self._tiles.append(self.build_tile())
        self.apply_pending_state()
        self.reflow_tiles()

    def build_tile(self) -> ChannelTile:
        root = QWidget(self._content)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        body = QHBoxLayout()
        root_layout.addLayout(body, 1)

        measure = TileMeasurement(
            root, self.measure_settings, lambda: self.tool, self.on_shape_changed
        )
        measure.apply_shape(self.shape)
        plot, image_item = self.build_plot(root, measure)
        body.addWidget(plot, 1)
        levels = LevelsControl(root, image_item, LUT)
        body.addWidget(levels)

        root_layout.addWidget(measure.strip)
        measure.show_hint_for(self.tool)
        return ChannelTile(root, plot, levels, measure)

    def build_plot(
        self, parent: QWidget, measure: TileMeasurement
    ) -> tuple[pg.PlotWidget, pg.ImageItem]:
        """A tile's plot and image, wired to pick the pixel under a click."""
        plot = pg.PlotWidget(parent, viewBox=measure.view_box)
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setAspectLocked(True)
        plot.invertY(True)
        image_item = pg.ImageItem()
        plot.addItem(image_item)
        scene = plot.sceneObj
        if scene is None:
            raise RuntimeError("a new PlotWidget has no scene")
        scene.sigMouseClicked.connect(lambda event: self.on_scene_clicked(event, image_item))
        return plot, image_item

    def on_scene_clicked(self, event: Any, image_item: pg.ImageItem) -> None:
        """Pick the pixel under a left click, unless a measuring tool is using the mouse."""
        if event.button() != Qt.MouseButton.LeftButton or self.tool is not None:
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
        program_output, image = self.shown_output, self.shown_image
        if program_output is None or image is None:
            return False
        _, height, width = image.shape
        if not (0 <= x < width and 0 <= y < height):
            return False
        self.inputs.set(PICK.key, PixelPick.on(program_output, x, y))
        return True

    def update_channel_image(self, index: int, channel: np.ndarray) -> None:
        tile = self._tiles[index]
        tile.levels.take(channel, autoscale_levels(channel))
        tile.measure.refresh(channel)

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
        return {"channels": [tile.levels.export_state().to_json() for tile in self._tiles]}

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        # Applied once tiles exist, since none do yet.
        self._pending_state = parse_levels(state.get("channels"))
        self.apply_pending_state()

    def apply_pending_state(self) -> None:
        """Apply saved tile settings, once."""
        # Once, so live edits during a run are not overwritten on every frame.
        if not self._pending_state or not self._tiles:
            return
        for tile, saved in zip(self._tiles, self._pending_state, strict=False):
            tile.levels.restore(saved)
        self._pending_state = []
