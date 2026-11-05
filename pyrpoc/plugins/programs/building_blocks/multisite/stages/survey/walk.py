"""One survey: tile by tile across the region, focusing where there is content, then
revisiting early tiles to measure motion, placing the tiles into an overview, and returning to
the start."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.plugins.program_io.inputs import FocusSearch
from pyrpoc.plugins.program_io.outputs import MosaicTile, MotionRate, SurveyFinding, SurveyTile

from ....focus.climb import climb
from ....focus.tenengrad import tenengrad
from ...analysis.content import content_channels, content_region
from ...analysis.motion import feature_speeds
from ...analysis.noise import MAD_TO_SIGMA, strongest_channel
from ...interfaces.field import FieldOfView
from ...interfaces.microscope import Microscope
from ...interfaces.progress import Progress
from .focus_surface import FocusSurface
from .grid import Tile, snake
from .layout import stitch
from .sink import SurveySink

# How many early tiles with content are revisited to measure motion.
REVISITS = 6


@dataclass(frozen=True)
class SurveySettings:
    """The region from the stage's corner, the field each tile is, the overlap, and how far
    focus is searched."""

    width_um: float
    height_um: float
    overlap: float
    field: FieldOfView
    search: FocusSearch


@dataclass(frozen=True)
class Taken:
    """One tile as taken: its dish-oriented frame, z, when, and which channels held content."""

    tile: Tile
    frame: np.ndarray
    z_um: float
    time_s: float
    channels: tuple[int, ...]


def motion_rate(speeds: list[float]) -> MotionRate | None:
    """The median speed and its robust spread, or None with nothing measured."""
    if not speeds:
        return None
    values = np.asarray(speeds)
    median = float(np.median(values))
    spread = float(np.median(np.abs(values - median))) * MAD_TO_SIGMA
    return MotionRate(median, spread, len(values))


class Surveying:
    """One survey in progress."""

    def __init__(
        self, progress: Progress, scope: Microscope, settings: SurveySettings, sink: SurveySink
    ):
        self.progress = progress
        self.scope = scope
        self.settings = settings
        self.sink = sink
        self.field = settings.field
        self.scan = settings.field.scan()
        self.start = scope.xy()
        self.surface = FocusSurface(scope.z(), settings.search.final_step_um)
        size = (settings.width_um, settings.height_um)
        self.tiles = snake(self.start, size, self.field.field_um, settings.overlap)
        grid = tuple(MosaicTile(t.row, t.col, t.x_um, t.y_um) for t in self.tiles)
        sink.plan(grid, self.field.um_per_px)
        self.taken: list[Taken] = []

    def frame(self) -> np.ndarray:
        return self.field.on_dish(self.scope.scan(self.scan, ()))

    def focus(self, guess: float, frame: np.ndarray, channels: tuple[int, ...]) -> float:
        """The sharpest z near ``guess``, judged where ``frame`` shows content."""
        weights = content_region(frame, channels)

        def measure(z_um: float) -> float:
            self.scope.move_z(z_um)
            return tenengrad(self.frame()[list(channels)], weights)

        search = self.surface.search(self.settings.search)
        return climb(measure, lambda _step: None, guess, search).best_z_um

    def take(self, tile: Tile) -> Taken:
        """Image one tile, focusing first if it holds anything."""
        self.scope.move_xy(tile.x_um, tile.y_um)
        z = self.surface.predict(tile.x_um, tile.y_um)
        self.scope.move_z(z)
        frame = self.frame()
        channels = content_channels(frame)
        if channels:
            self.progress.status(f"tile {tile.index + 1}/{len(self.tiles)}: focusing")
            z = self.focus(z, frame, channels)
            self.surface.add(tile.x_um, tile.y_um, z)
            self.scope.move_z(z)
            frame = self.frame()
        return Taken(tile, frame, z, self.scope.now(), channels)

    def walk(self) -> None:
        for tile in self.tiles:
            self.progress.check_cancel()
            self.progress.status(f"tile {tile.index + 1}/{len(self.tiles)}")
            taken = self.take(tile)
            self.taken.append(taken)
            how = SurveyTile(taken.z_um, taken.time_s, bool(taken.channels))
            self.sink.add_tile(taken.frame, self.scope.channel_labels, how)

    def revisit(self) -> MotionRate | None:
        """Find the features of the earliest tiles with content again, now."""
        speeds: list[float] = []
        early = [taken for taken in self.taken if taken.channels][:REVISITS]
        for number, taken in enumerate(early, start=1):
            self.progress.status(f"measuring motion {number}/{len(early)}")
            self.scope.move_xy(taken.tile.x_um, taken.tile.y_um)
            self.scope.move_z(taken.z_um)
            now = self.frame()
            minutes = (self.scope.now() - taken.time_s) / 60.0
            channel = taken.channels[strongest_channel(taken.frame[list(taken.channels)])]
            speeds += feature_speeds(
                taken.frame[channel], now[channel], minutes, self.field.um_per_px
            )
        return motion_rate(speeds)

    def finish(self, motion: MotionRate | None) -> None:
        """Place the tiles into the overview, hand it and what was found to the sink, and go
        back to the start."""
        self.progress.status("stitching")
        corners = [self.field.corner_um(t.tile.x_um, t.tile.y_um) for t in self.taken]
        frames = [taken.frame for taken in self.taken]
        times = [taken.time_s for taken in self.taken]
        rate = 0.0 if motion is None else motion.um_per_min
        overview, placed = stitch(frames, corners, times, self.tiles, self.field.um_per_px, rate)
        self.sink.finish(SurveyFinding(self.surface.plane, motion, tuple(placed)), overview)
        self.scope.move_xy(*self.start)
        self.scope.move_z(self.surface.predict(*self.start))


def survey(
    progress: Progress, scope: Microscope, settings: SurveySettings, sink: SurveySink
) -> None:
    """Walk the region, measure motion, place the tiles, and fill ``sink``."""
    surveying = Surveying(progress, scope, settings, sink)
    try:
        surveying.walk()
        surveying.finish(surveying.revisit())
    except BaseException:
        scope.stop()
        raise
