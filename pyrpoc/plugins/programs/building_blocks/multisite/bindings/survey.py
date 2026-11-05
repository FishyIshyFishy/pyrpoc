"""The survey stage bound to a run: its program inputs, what it reads and fills, and the call
a program makes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pyrpoc.core.program import RunContext
from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.plugins.program_io.inputs import (
    DWELL_TIME_US,
    FIELD_PIXELS,
    FIELD_UM,
    FOCUS_FINAL_STEP_UM,
    FOCUS_INITIAL_STEP_UM,
    FOCUS_SEARCH_RANGE_UM,
    SURVEY_HEIGHT_UM,
    SURVEY_WIDTH_UM,
    TILE_OVERLAP,
    FocusSearch,
)
from pyrpoc.plugins.program_io.outputs import (
    GalvoCalibration,
    GalvoCalibrationOutput,
    MosaicTile,
    PlacedImageOutput,
    SurveyFinding,
    SurveyOutput,
    SurveyTile,
)

from ..interfaces.field import FieldOfView
from ..interfaces.microscope import Microscope
from ..interfaces.overview import Overview
from ..stages.survey.walk import SurveySettings, survey

REGION_INPUTS = (SURVEY_WIDTH_UM, SURVEY_HEIGHT_UM)
TILE_INPUTS = (FIELD_UM, FIELD_PIXELS, TILE_OVERLAP, DWELL_TIME_US)
SURVEY_FOCUS_INPUTS = (FOCUS_SEARCH_RANGE_UM, FOCUS_INITIAL_STEP_UM, FOCUS_FINAL_STEP_UM)
# What every survey fills: its tiles, and the overview placed from them.
SURVEY_EMITS: dict[str, type[ProgramOutput]] = {
    "tiles": SurveyOutput,
    "overview": PlacedImageOutput,
}
# The calibration its field is scanned with.
SURVEY_READS: dict[str, type[ProgramOutput]] = {"calibration": GalvoCalibrationOutput}


def settings_from(
    values: ProgramInputValues, calibration: GalvoCalibration, overscan: tuple[int, int]
) -> SurveySettings:
    field = FieldOfView(
        values[FIELD_UM],
        values[FIELD_PIXELS],
        (calibration.um_per_volt_x, calibration.um_per_volt_y),
        values[DWELL_TIME_US],
        *overscan,
    )
    return SurveySettings(
        values[SURVEY_WIDTH_UM],
        values[SURVEY_HEIGHT_UM],
        values[TILE_OVERLAP],
        field,
        FocusSearch.from_inputs(values),
    )


class SurveyOutputs:
    """A run's ``tiles`` and ``overview`` outputs, as the survey's sink."""

    def __init__(self, ctx: RunContext):
        self.tiles = ctx.output("tiles", SurveyOutput)
        self.overview = ctx.output("overview", PlacedImageOutput)

    def plan(self, tiles: tuple[MosaicTile, ...], um_per_px: float) -> None:
        self.tiles.plan_survey(tiles, um_per_px)

    def add_tile(self, frame: np.ndarray, channels: Sequence[str], taken: SurveyTile) -> None:
        self.tiles.add_tile(frame, channels, taken)

    def finish(self, finding: SurveyFinding, overview: Overview) -> None:
        # The overview first, so a finished survey always has its overview beside it.
        self.overview.add_placed(overview.planes, self.tiles.channels, overview.placement)
        self.tiles.finish(finding)


def run_survey(ctx: RunContext, scope: Microscope, overscan: tuple[int, int]) -> None:
    """Survey the region from the stage's corner, with the calibration read."""
    calibration = ctx.earlier("calibration", GalvoCalibrationOutput).measured()
    settings = settings_from(ctx.inputs, calibration, overscan)
    survey(ctx, scope, settings, SurveyOutputs(ctx))
