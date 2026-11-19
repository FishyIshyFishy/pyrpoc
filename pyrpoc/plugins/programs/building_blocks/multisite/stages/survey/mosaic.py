"""One mosaic: a grid of tiles centred where the stage is, each imaged where it stands with no
focusing and blended into the overview as it is taken, then the stage returned to the start."""

# Each tile is placed where the stage put it, at the calibration's pixel size: nothing is
# registered from the images, since a mosaic measures no motion or drift to correct for.

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pyrpoc.plugins.program_io.outputs import (
    MosaicTile,
    Placement,
    SurveyEnded,
    TileTaken,
    TileVisiting,
)

from ...analysis.content import content_channels
from ...analysis.stitching import Blend
from ...interfaces.cancellation import Cancellation
from ...interfaces.field import FieldOfView
from ...interfaces.microscope import Microscope
from ...interfaces.overview import Overview
from .grid import Tile, centred
from .layout import OVERVIEW_PX
from .sink import MosaicSink


@dataclass(frozen=True)
class MosaicSettings:
    """How many tiles along x and y, the field each tile is, and the share neighbours
    overlap."""

    columns: int
    rows: int
    overlap: float
    field: FieldOfView


class Placing:
    """Where every tile of the grid lies in the overview, and the overview so far."""

    def __init__(self, tiles: tuple[Tile, ...], field: FieldOfView, channels: int):
        self.field = field
        corners = [field.corner_um(tile.x_um, tile.y_um) for tile in tiles]
        self.left_um = min(left for left, _top in corners)
        self.top_um = min(top for _left, top in corners)
        reach = (
            (max(top for _left, top in corners) - self.top_um) / field.um_per_px,
            (max(left for left, _top in corners) - self.left_um) / field.um_per_px,
        )
        # The overview's longest side is kept to OVERVIEW_PX; a larger mosaic is shrunk to fit.
        self.factor = max(1, math.ceil((max(reach) + field.pixels) / OVERVIEW_PX))
        self.blend = Blend(channels, (field.pixels, field.pixels), reach, self.factor)

    def add(self, tile: Tile, frame: np.ndarray) -> None:
        left, top = self.field.corner_um(tile.x_um, tile.y_um)
        um = self.field.um_per_px
        self.blend.add(frame, (top - self.top_um) / um, (left - self.left_um) / um)

    def overview(self) -> Overview:
        placement = Placement(self.left_um, self.top_um, self.field.um_per_px * self.factor)
        return Overview(self.blend.image(), placement)


def mosaic(
    cancellation: Cancellation, scope: Microscope, settings: MosaicSettings, sink: MosaicSink
) -> None:
    """Image the grid around the stage, blending each tile in as it is taken, and fill
    ``sink``."""
    field = settings.field
    scan = field.scan()
    start = scope.xy()
    tiles = centred(start, settings.columns, settings.rows, field.field_um, settings.overlap)
    sink.plan(tuple(MosaicTile(t.row, t.col, t.x_um, t.y_um) for t in tiles), field.um_per_px)
    placing = Placing(tiles, field, len(scope.channel_labels))
    try:
        for tile in tiles:
            cancellation.check_cancel()
            sink.log(TileVisiting(scope.now(), tile.index, len(tiles)))
            scope.move_xy(tile.x_um, tile.y_um)
            frame = field.on_dish(scope.scan(scan, ()))
            placing.add(tile, frame)
            sink.add_tile(frame, scope.channel_labels)
            content = bool(content_channels(frame))
            sink.log(TileTaken(scope.now(), tile.index, len(tiles), scope.z(), content))
        sink.finish(placing.overview())
        scope.move_xy(*start)
    except BaseException:
        scope.stop()
        raise
    sink.log(SurveyEnded(scope.now()))
