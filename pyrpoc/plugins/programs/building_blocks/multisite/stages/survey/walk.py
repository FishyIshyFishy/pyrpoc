"""One survey: tile by tile across the region, focusing where there is content, then
revisiting early tiles to measure motion, placing the tiles into an overview, and returning to
the start."""

# Focus is predicted from the plane through the tiles focused so far, and searched for only as far
# as that plane is unsure. Two things say the prediction was wrong, and each widens the search to
# the whole range: a search that ends at its own edge (focus lay beyond it), and a tile with
# nothing in view beside tiles that held content (it is likelier out of focus than empty). Without
# the second, a survey that lost focus would see only empty tiles, which are never focused, and
# never find it again.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.plugins.program_io.inputs import FocusSearch
from pyrpoc.plugins.program_io.outputs import (
    FocusWidened,
    MosaicTile,
    MotionMeasured,
    MotionRate,
    MotionRevisiting,
    Stitching,
    SurveyEnded,
    SurveyFinding,
    SurveyTile,
    TileFocusing,
    TileTaken,
    TileVisiting,
)

from ....focus.climb import ClimbResult, climb, sweep_then_climb
from ....focus.tenengrad import tenengrad
from ...analysis.content import content_channels, content_region
from ...analysis.motion import feature_speeds
from ...analysis.noise import MAD_TO_SIGMA, strongest_channel
from ...interfaces.cancellation import Cancellation
from ...interfaces.field import FieldOfView
from ...interfaces.microscope import Microscope
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
        self,
        cancellation: Cancellation,
        scope: Microscope,
        settings: SurveySettings,
        sink: SurveySink,
    ):
        self.cancellation = cancellation
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

    def climb_from(
        self, guess: float, frame: np.ndarray, channels: tuple[int, ...], search: FocusSearch
    ) -> ClimbResult:
        """The sharpest z within ``search`` of ``guess``, judged where ``frame`` shows
        content."""
        weights = content_region(frame, channels)

        def measure(z_um: float) -> float:
            self.scope.move_z(z_um)
            return tenengrad(self.frame()[list(channels)], weights)

        # A range wider than one first step is swept before it is climbed, since the climb
        # only finds a peak it starts within a step of.
        wide = search.search_range_um > search.initial_step_um
        how = sweep_then_climb if wide else climb
        return how(measure, lambda _step: None, guess, search)

    def focus(
        self, tile: Tile, guess: float, frame: np.ndarray, channels: tuple[int, ...], wide: bool
    ) -> tuple[float, bool]:
        """The sharpest z near ``guess``, searched for as far as the plane is unsure at
        ``tile`` (or over the whole range when ``wide``), and whether it is a measurement of
        focus rather than the edge of the range."""
        full = self.settings.search
        search = full if wide else self.surface.search(full, tile.x_um, tile.y_um)
        found = self.climb_from(guess, frame, channels, search)
        if found.at_edge and search != full:
            self.log_widened(tile, "edge")
            found = self.climb_from(guess, frame, channels, full)
        return found.best_z_um, not found.at_edge

    def log_widened(self, tile: Tile, reason: str) -> None:
        self.sink.log(FocusWidened(self.scope.now(), tile.index, len(self.tiles), reason))

    def beside_content(self, tile: Tile) -> bool:
        """Whether a tile already taken next to ``tile`` held content."""
        return any(
            taken.channels
            and abs(taken.tile.row - tile.row) <= 1
            and abs(taken.tile.col - tile.col) <= 1
            for taken in self.taken
        )

    def find_content(self, guess: float) -> tuple[float, np.ndarray, tuple[int, ...]] | None:
        """Stepping out from ``guess`` by the search's first step, either way and as far as
        its range: the first z at which anything is in view, its frame and the channels
        holding it; None if nowhere."""
        full = self.settings.search
        steps = int(full.search_range_um // full.initial_step_um)
        for count in range(1, steps + 1):
            for side in (1.0, -1.0):
                z = guess + side * count * full.initial_step_um
                self.cancellation.check_cancel()
                self.scope.move_z(z)
                frame = self.frame()
                channels = content_channels(frame)
                if channels:
                    return z, frame, channels
        return None

    def take(self, tile: Tile) -> Taken:
        """Image one tile, focusing first if it holds anything."""
        self.sink.log(TileVisiting(self.scope.now(), tile.index, len(self.tiles)))
        self.scope.move_xy(tile.x_um, tile.y_um)
        predicted = z = self.surface.predict(tile.x_um, tile.y_um)
        self.scope.move_z(z)
        frame = self.frame()
        channels = content_channels(frame)
        wide = False
        if not channels and self.beside_content(tile):
            self.log_widened(tile, "lost")
            found = self.find_content(predicted)
            if found is not None:
                z, frame, channels = found
                wide = True
            else:
                self.scope.move_z(predicted)
                frame = self.frame()
        if channels:
            self.sink.log(TileFocusing(self.scope.now(), tile.index, len(self.tiles)))
            # Climbed from wherever content came into view: the prediction, or the z the
            # search for content found it at.
            z, measured = self.focus(tile, z, frame, channels, wide)
            if measured:
                self.surface.add(tile.x_um, tile.y_um, z)
            self.scope.move_z(z)
            frame = self.frame()
        return Taken(tile, frame, z, self.scope.now(), channels)

    def walk(self) -> None:
        for tile in self.tiles:
            self.cancellation.check_cancel()
            taken = self.take(tile)
            self.taken.append(taken)
            content = bool(taken.channels)
            how = SurveyTile(taken.z_um, taken.time_s, content)
            self.sink.add_tile(taken.frame, self.scope.channel_labels, how)
            done = TileTaken(taken.time_s, tile.index, len(self.tiles), taken.z_um, content)
            self.sink.log(done)

    def revisit(self) -> MotionRate | None:
        """Find the features of the earliest tiles with content again, now."""
        speeds: list[float] = []
        early = [taken for taken in self.taken if taken.channels][:REVISITS]
        for number, taken in enumerate(early):
            self.sink.log(MotionRevisiting(self.scope.now(), number, len(early)))
            self.scope.move_xy(taken.tile.x_um, taken.tile.y_um)
            self.scope.move_z(taken.z_um)
            now = self.frame()
            minutes = (self.scope.now() - taken.time_s) / 60.0
            channel = taken.channels[strongest_channel(taken.frame[list(taken.channels)])]
            speeds += feature_speeds(
                taken.frame[channel], now[channel], minutes, self.field.um_per_px
            )
        motion = motion_rate(speeds)
        rate = None if motion is None else motion.um_per_min
        self.sink.log(MotionMeasured(self.scope.now(), rate, len(speeds)))
        return motion

    def finish(self, motion: MotionRate | None) -> None:
        """Place the tiles into the overview, hand it and what was found to the sink, and go
        back to the start."""
        self.sink.log(Stitching(self.scope.now()))
        corners = [self.field.corner_um(t.tile.x_um, t.tile.y_um) for t in self.taken]
        frames = [taken.frame for taken in self.taken]
        times = [taken.time_s for taken in self.taken]
        rate = 0.0 if motion is None else motion.um_per_min
        overview, placed = stitch(frames, corners, times, self.tiles, self.field.um_per_px, rate)
        self.sink.finish(SurveyFinding(self.surface.plane, motion, tuple(placed)), overview)
        self.scope.move_xy(*self.start)
        self.scope.move_z(self.surface.predict(*self.start))
        self.sink.log(SurveyEnded(self.scope.now()))


def survey(
    cancellation: Cancellation, scope: Microscope, settings: SurveySettings, sink: SurveySink
) -> None:
    """Walk the region, measure motion, place the tiles, and fill ``sink``."""
    surveying = Surveying(cancellation, scope, settings, sink)
    try:
        surveying.walk()
        surveying.finish(surveying.revisit())
    except BaseException:
        scope.stop()
        raise
