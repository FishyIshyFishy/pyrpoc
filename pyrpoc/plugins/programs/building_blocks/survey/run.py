"""One survey: tile by tile across the region, focusing where there is content, then
revisiting early tiles to measure motion, stitching, and returning to the start."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.core.program import EarlierRun, RunContext
from pyrpoc.core.program_input import ParameterError, ProgramInputValues
from pyrpoc.plugins.program_io.inputs import (
    DWELL_TIME_US,
    FIELD_PIXELS,
    FIELD_UM,
    SURVEY_HEIGHT_UM,
    SURVEY_WIDTH_UM,
    TILE_OVERLAP,
    FocusSearch,
)
from pyrpoc.plugins.program_io.outputs import (
    GalvoCalibration,
    Image,
    Mosaic,
    MotionRate,
    PlacedTile,
    SurveyResult,
    SurveyTile,
)

from ..field import Field
from ..focus.climb import climb
from ..focus.tenengrad import tenengrad
from ..microscope import Microscope
from ..vision.noise import strongest_channel
from .content import content_channels, content_region
from .focus_surface import FocusSurface
from .grid import Tile, snake
from .motion import feature_speeds, motion_rate
from .stitch import stitch

# How many early tiles with content are revisited to measure motion.
REVISITS = 6


def calibration_from(ctx: RunContext, run: EarlierRun) -> GalvoCalibration:
    """The galvo calibration the run was handed, read from a recording on disk."""
    value = ctx.earlier(run)["calibration"].value
    if not isinstance(value, GalvoCalibration):
        raise ParameterError("the Galvo Calibration in this folder holds no result")
    return value


@dataclass(frozen=True)
class SurveySettings:
    """The region from the stage's corner, the field each tile is, the overlap, and how far
    focus is searched."""

    width_um: float
    height_um: float
    overlap: float
    field: Field
    search: FocusSearch

    @classmethod
    def from_inputs(
        cls, values: ProgramInputValues, calibration: GalvoCalibration, overscan: tuple[int, int]
    ) -> SurveySettings:
        field = Field(
            values[FIELD_UM],
            values[FIELD_PIXELS],
            (calibration.um_per_volt_x, calibration.um_per_volt_y),
            values[DWELL_TIME_US],
            *overscan,
        )
        return cls(
            values[SURVEY_WIDTH_UM],
            values[SURVEY_HEIGHT_UM],
            values[TILE_OVERLAP],
            field,
            FocusSearch.from_inputs(values),
        )


@dataclass(frozen=True)
class Taken:
    """One tile as taken: its dish-oriented frame, z, when, and which channels held content."""

    tile: Tile
    frame: np.ndarray
    z_um: float
    time_s: float
    channels: tuple[int, ...]


class Surveying:
    """One survey in progress."""

    def __init__(self, ctx: RunContext, scope: Microscope, settings: SurveySettings):
        self.ctx = ctx
        self.scope = scope
        self.settings = settings
        self.field = settings.field
        self.scan = settings.field.scan()
        self.start = scope.xy()
        self.surface = FocusSurface(scope.z(), settings.search.final_step_um)
        size = (settings.width_um, settings.height_um)
        self.tiles = snake(self.start, size, self.field.field_um, settings.overlap)
        self.mosaic = Mosaic(self.field.um_per_px, len(self.tiles), (), None)
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
            self.ctx.status(f"tile {tile.index + 1}/{len(self.tiles)}: focusing")
            z = self.focus(z, frame, channels)
            self.surface.add(tile.x_um, tile.y_um, z)
            self.scope.move_z(z)
            frame = self.frame()
        return Taken(tile, frame, z, self.scope.now(), channels)

    def publish(self, taken: Taken) -> None:
        left, top = self.field.corner_um(taken.tile.x_um, taken.tile.y_um)
        image = Image(taken.frame, self.scope.channel_labels)
        self.mosaic = self.mosaic.with_tile(PlacedTile(image, left, top))
        self.ctx.publish("mosaic", self.mosaic)

    def walk(self) -> None:
        for tile in self.tiles:
            self.ctx.check_cancel()
            self.ctx.status(f"tile {tile.index + 1}/{len(self.tiles)}")
            taken = self.take(tile)
            self.taken.append(taken)
            self.publish(taken)

    def revisit(self) -> MotionRate | None:
        """Find the features of the earliest tiles with content again, now."""
        speeds: list[float] = []
        early = [taken for taken in self.taken if taken.channels][:REVISITS]
        for number, taken in enumerate(early, start=1):
            self.ctx.status(f"measuring motion {number}/{len(early)}")
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
        """Stitch, publish what was found, and go back to the start."""
        self.ctx.status("stitching")
        corners = [self.field.corner_um(t.tile.x_um, t.tile.y_um) for t in self.taken]
        frames = [taken.frame for taken in self.taken]
        times = [taken.time_s for taken in self.taken]
        rate = 0.0 if motion is None else motion.um_per_min
        overview, placed = stitch(
            frames,
            corners,
            times,
            self.tiles,
            self.field.um_per_px,
            rate,
            self.scope.channel_labels,
        )
        self.ctx.publish("mosaic", self.mosaic.with_overview(overview))
        self.ctx.publish("survey", self.result(motion, placed))
        self.scope.move_xy(*self.start)
        self.scope.move_z(self.surface.predict(*self.start))

    def result(self, motion: MotionRate | None, placed: list[tuple[float, float]]) -> SurveyResult:
        tiles = tuple(
            SurveyTile(
                t.tile.x_um,
                t.tile.y_um,
                t.z_um,
                t.time_s,
                bool(t.channels),
                left,
                top,
            )
            for t, (left, top) in zip(self.taken, placed, strict=True)
        )
        settings, field = self.settings, self.field
        return SurveyResult(
            self.start[0],
            self.start[1],
            settings.width_um,
            settings.height_um,
            field.field_um,
            field.pixels,
            field.um_per_volt[0],
            field.um_per_volt[1],
            self.surface.plane,
            motion,
            tiles,
        )


def survey(ctx: RunContext, scope: Microscope, settings: SurveySettings) -> None:
    """Walk the region, measure motion, stitch, and publish the mosaic and the result."""
    surveying = Surveying(ctx, scope, settings)
    try:
        surveying.walk()
        surveying.finish(surveying.revisit())
    except BaseException:
        scope.stop()
        raise
