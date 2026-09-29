"""One tile per channel, with a per-tile name, autoscale and LUT. Clicking a
pixel answers a ``PixelPick`` request."""

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
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.structs.data import Image2D, Library
from pyrpoc.structs.picks import Pick, PixelPick

from ..components.dataset_panel import DatasetPanel, panel_registry
from ..components.levels import autoscale_levels, mono_levels

LUT = pg.ColorMap(
    pos=np.array([0.0, 0.999, 1.0], dtype=float),
    color=np.array([[0, 0, 0, 255], [255, 255, 255, 255], [255, 0, 0, 255]], dtype=np.ubyte),
)


@dataclass
class ChannelTile:
    root: QWidget
    name_edit: QLineEdit
    autoscale_box: QCheckBox
    plot: pg.PlotWidget
    image_item: pg.ImageItem
    hist_widget: pg.HistogramLUTWidget
    min_val: float = 0.0
    max_val: float = 1.0


@dataclass
class TileState:
    """One tile's saved settings, parsed from the workspace."""

    index: int
    name: str
    autoscale: bool
    min_val: float
    max_val: float


@panel_registry.register("image_2d")
class Image2DPanel(DatasetPanel):
    display_name = "2D Tiled"
    renders = [Image2D]

    def __init__(self, library: Library):
        super().__init__(library)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._tiles: list[ChannelTile] = []
        self._pending_state: list[TileState] = []
        self._suspend_lut_signal = False
        # Held here rather than per tile because tiles come and go with the
        # channel count.
        self._picking = False

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

    def refresh(self) -> None:
        dataset = self.dataset()
        frame = dataset.latest() if dataset is not None else None
        if dataset is None or frame is None:
            self.sync_channel_tiles(0)
            return
        arr = np.asarray(frame, dtype=np.float32)
        self.sync_channel_tiles(arr.shape[0])
        self.apply_channel_names(dataset.channel_labels)
        for index in range(arr.shape[0]):
            self.update_channel_image(index, arr[index])

    def apply_channel_names(self, labels: list[str]) -> None:
        """Name a tile after its acquisition channel, unless renamed by hand."""
        for index, (tile, label) in enumerate(zip(self._tiles, labels, strict=False)):
            if tile.name_edit.text().strip() in ("", f"Input {index + 1}"):
                tile.name_edit.blockSignals(True)
                tile.name_edit.setText(label)
                tile.name_edit.blockSignals(False)

    def sync_channel_tiles(self, count: int) -> None:
        while len(self._tiles) > count:
            tile = self._tiles.pop()
            tile.root.setParent(None)
            tile.root.deleteLater()
        while len(self._tiles) < count:
            self._tiles.append(self.build_tile(len(self._tiles)))
        self.apply_pending_state()
        self.reflow_tiles()

    def build_tile(self, index: int) -> ChannelTile:
        root = QWidget(self._content)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        name_edit = QLineEdit(f"Input {index + 1}", root)
        root_layout.addWidget(name_edit)
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

        tile = ChannelTile(root, name_edit, autoscale_box, plot, image_item, hist_widget)
        autoscale_box.toggled.connect(lambda _checked, i=index: self.on_autoscale_toggled(i))
        hist_widget.item.sigLevelsChanged.connect(
            lambda _item, i=index: self.on_lut_levels_changed(i)
        )
        self.apply_pick_cursor(tile)
        return tile

    def build_plot(self, parent: QWidget) -> tuple[pg.PlotWidget, pg.ImageItem]:
        """The image and its click wiring. Wired at build rather than when a pick
        is requested, because a tile can appear after arming."""
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

    def set_pick_mode(self, kind: type[Pick] | None) -> None:
        """Only a request a ``PixelPick`` satisfies turns the crosshair on."""
        self._picking = kind is not None and issubclass(PixelPick, kind)
        for tile in self._tiles:
            self.apply_pick_cursor(tile)

    def apply_pick_cursor(self, tile: ChannelTile) -> None:
        """Crosshair over the plot only: it means "clicking here moves
        hardware", so it must not appear where a click does nothing."""
        if self._picking:
            tile.plot.setCursor(Qt.CursorShape.CrossCursor)
        else:
            tile.plot.unsetCursor()

    def on_scene_clicked(self, event: Any, image_item: pg.ImageItem) -> None:
        """Report the pixel under a left click. Bounds are checked against the
        frame: a click just outside the pixels is still inside the view box,
        and an out-of-range index would park the galvos outside the scan."""
        if not self._picking or event.button() != Qt.MouseButton.LeftButton:
            return
        dataset = self.dataset()
        frame = dataset.latest() if dataset is not None else None
        if dataset is None or frame is None:
            return
        point = image_item.mapFromScene(event.scenePos())
        x, y = int(np.floor(point.x())), int(np.floor(point.y()))
        if not (0 <= x < frame.shape[2] and 0 <= y < frame.shape[1]):
            return
        event.accept()
        self.picked.emit(PixelPick(dataset, x, y))

    def on_autoscale_toggled(self, index: int) -> None:
        frame = self.latest()
        if frame is not None:
            self.update_channel_image(index, np.asarray(frame[index], dtype=np.float32))

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

        columns = 2
        rows = (len(self._tiles) + columns - 1) // columns
        for index, tile in enumerate(self._tiles):
            row, col = index // columns, index % columns
            self._grid.addWidget(tile.root, row, col)
            # Keep all tiles evenly sized as the channel count grows.
            self._grid.setRowStretch(row, 1)
            self._grid.setColumnStretch(col, 1)
        # Clear stale stretch left by removed rows.
        for row in range(rows, max(rows, 8)):
            self._grid.setRowStretch(row, 0)

    def export_persistence_state(self) -> dict[str, Any]:
        return {
            "channels": [
                {
                    "index": index,
                    "name": tile.name_edit.text().strip(),
                    "autoscale": tile.autoscale_box.isChecked(),
                    "min_val": tile.min_val,
                    "max_val": tile.max_val,
                }
                for index, tile in enumerate(self._tiles)
            ]
        }

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        """Parse saved tile settings from workspace JSON, a boundary: malformed
        rows are skipped. Applied once tiles exist, since none do yet."""
        channels = state.get("channels", [])
        if not isinstance(channels, list):
            return
        self._pending_state = [
            TileState(
                index=int(row.get("index", position)),
                name=str(row.get("name", "")).strip(),
                autoscale=bool(row.get("autoscale", True)),
                min_val=float(row.get("min_val", 0.0)),
                max_val=float(row.get("max_val", 1.0)),
            )
            for position, row in enumerate(channels)
            if isinstance(row, dict)
        ]
        self.apply_pending_state()

    def apply_pending_state(self) -> None:
        """Apply saved settings once, so live edits during a run are not
        overwritten on every frame."""
        if not self._pending_state or not self._tiles:
            return
        for saved in self._pending_state:
            if not 0 <= saved.index < len(self._tiles):
                continue
            tile = self._tiles[saved.index]
            tile.name_edit.blockSignals(True)
            tile.name_edit.setText(saved.name or f"Input {saved.index + 1}")
            tile.name_edit.blockSignals(False)
            tile.autoscale_box.blockSignals(True)
            tile.autoscale_box.setChecked(saved.autoscale)
            tile.autoscale_box.blockSignals(False)
            self.apply_levels(tile, saved.min_val, max(saved.max_val, saved.min_val + 1e-12))
        self._pending_state = []
