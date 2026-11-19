"""One mosaic: a grid of tiles centred where the stage is, each imaged where it stands with no
focusing, placed into an overview by where the stage put them and what they share, and the stage
returned to the start."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.plugins.program_io.outputs import (
    MosaicTile,
    Stitching,
    SurveyEnded,
    TileTaken,
    TileVisiting,
)

from ...analysis.content import content_channels
from ...interfaces.cancellation import Cancellation
from ...interfaces.field import FieldOfView
from ...interfaces.microscope import Microscope
from .grid import centred
from .layout import stitch
from .sink import MosaicSink


@dataclass(frozen=True)
class MosaicSettings:
    """How many tiles along x and y, the field each tile is, and the share neighbours
    overlap."""

    columns: int
    rows: int
    overlap: float
    field: FieldOfView


def mosaic(
    cancellation: Cancellation, scope: Microscope, settings: MosaicSettings, sink: MosaicSink
) -> None:
    """Image the grid around the stage, place the tiles, and fill ``sink``."""
    field = settings.field
    scan = field.scan()
    start = scope.xy()
    tiles = centred(start, settings.columns, settings.rows, field.field_um, settings.overlap)
    sink.plan(tuple(MosaicTile(t.row, t.col, t.x_um, t.y_um) for t in tiles), field.um_per_px)
    frames, times = [], []
    try:
        for tile in tiles:
            cancellation.check_cancel()
            sink.log(TileVisiting(scope.now(), tile.index, len(tiles)))
            scope.move_xy(tile.x_um, tile.y_um)
            frame = field.on_dish(scope.scan(scan, ()))
            frames.append(frame)
            times.append(scope.now())
            sink.add_tile(frame, scope.channel_labels)
            content = bool(content_channels(frame))
            sink.log(TileTaken(times[-1], tile.index, len(tiles), scope.z(), content))
        sink.log(Stitching(scope.now()))
        corners = [field.corner_um(tile.x_um, tile.y_um) for tile in tiles]
        # Nothing is taken to move between tiles: a mosaic is quick, and unlike a survey it
        # measures no motion to allow for.
        overview, _placed = stitch(frames, corners, times, tiles, field.um_per_px, 0.0)
        sink.finish(overview)
        scope.move_xy(*start)
    except BaseException:
        scope.stop()
        raise
    sink.log(SurveyEnded(scope.now()))
