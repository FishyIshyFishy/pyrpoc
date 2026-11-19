"""The mosaic bound to a run: its program inputs, what it reads and fills, and the call a
program makes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pyrpoc.core.program import RunContext
from pyrpoc.core.program_input import InputReader, ProgramInput
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.plugins.program_io.inputs import (
    CALIBRATION,
    DWELL_TIME_US,
    FIELD_PIXELS,
    FIELD_UM,
    TILE_OVERLAP,
    X_TILES,
    Y_TILES,
)
from pyrpoc.plugins.program_io.outputs import (
    GalvoCalibration,
    MosaicOutput,
    MosaicTile,
    SurveyEvent,
    SurveyLogOutput,
)

from ..interfaces.field import FieldOfView
from ..interfaces.microscope import Microscope
from ..interfaces.overview import Overview
from ..stages.survey.mosaic import MosaicSettings, mosaic
from .earlier import required

GRID_INPUTS = (X_TILES, Y_TILES)
MOSAIC_TILE_INPUTS = (FIELD_UM, FIELD_PIXELS, TILE_OVERLAP, DWELL_TIME_US)
# What every mosaic fills: its tiles and the overview made of them, and what it did.
MOSAIC_EMITS: dict[str, type[ProgramOutput]] = {"mosaic": MosaicOutput, "log": SurveyLogOutput}
# The calibration its field is scanned and its tiles placed with, the newest in its folder.
MOSAIC_READS: list[ProgramInput[Any]] = [CALIBRATION]


def settings_from(
    values: InputReader, calibration: GalvoCalibration, overscan: tuple[int, int]
) -> MosaicSettings:
    field = FieldOfView(
        values[FIELD_UM],
        values[FIELD_PIXELS],
        (calibration.um_per_volt_x, calibration.um_per_volt_y),
        values[DWELL_TIME_US],
        *overscan,
    )
    return MosaicSettings(values[X_TILES], values[Y_TILES], values[TILE_OVERLAP], field)


class MosaicOutputs:
    """A run's ``mosaic`` and ``log`` outputs, as the mosaic's sink. Each event is also the
    run's status."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.tiles = ctx.output("mosaic", MosaicOutput)
        self.events = ctx.output("log", SurveyLogOutput)

    def log(self, event: SurveyEvent) -> None:
        self.events.log(event)
        self.ctx.status(event.text())

    def plan(self, tiles: tuple[MosaicTile, ...], um_per_px: float) -> None:
        self.tiles.plan(tiles, um_per_px)

    def add_tile(self, frame: np.ndarray, channels: Sequence[str]) -> None:
        self.tiles.add(frame, channels)

    def finish(self, overview: Overview) -> None:
        self.tiles.stitched(overview.planes, overview.placement)


def run_mosaic(ctx: RunContext, scope: Microscope, overscan: tuple[int, int]) -> None:
    """Image the grid around the stage, with the calibration read."""
    calibration = required(ctx, CALIBRATION).measured()
    mosaic(ctx, scope, settings_from(ctx.inputs, calibration, overscan), MosaicOutputs(ctx))
