"""The loop stage bound to a run: its program inputs, what it reads and fills, how it is
steered, and the call a program makes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pyrpoc.core.program import RunContext
from pyrpoc.core.program_input import ParameterError, ProgramInput
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.plugins.program_io.inputs import (
    DWELL_TIME_US,
    FOCUS_CHECK_S,
    FOCUS_FINAL_STEP_UM,
    PAUSED_UNITS,
    UNITS,
    Unit,
)
from pyrpoc.plugins.program_io.outputs import (
    FrameTag,
    GalvoCalibrationOutput,
    LoopEvent,
    LoopLogOutput,
    PlacedImageOutput,
    SurveyOutput,
    UnitFramesOutput,
)
from pyrpoc.plugins.program_io.requests import RUN_NOW

from ..interfaces.field import FieldOfView
from ..interfaces.microscope import Microscope
from ..interfaces.overview import Overview
from ..stages.loop.orchestrator import LoopSetup, Orchestrator, check_units
from ..stages.loop.routine import RoutineSpec
from ..stages.loop.survey_map import SurveyMap, Visit

# Every loop program declares these live, and accepts RUN_NOW, so the units can be steered.
STEERING: list[ProgramInput[Any]] = [UNITS, PAUSED_UNITS]
LOOP_FOCUS_INPUTS = (FOCUS_CHECK_S, FOCUS_FINAL_STEP_UM)
# What every loop fills: its log, and every frame of every unit.
LOOP_EMITS: dict[str, type[ProgramOutput]] = {"log": LoopLogOutput, "frames": UnitFramesOutput}
# What every loop reads: the survey, its overview, and the calibration its scans use.
LOOP_READS: dict[str, type[ProgramOutput]] = {
    "survey": SurveyOutput,
    "overview": PlacedImageOutput,
    "calibration": GalvoCalibrationOutput,
}


class RunSteering:
    """A run's live units and pauses, and the run-now requests posted to it."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    def units(self) -> tuple[Unit, ...]:
        return self.ctx.current(UNITS)

    def paused(self) -> frozenset[int]:
        return frozenset(self.ctx.current(PAUSED_UNITS))

    def run_now(self) -> list[int]:
        return self.ctx.requests(RUN_NOW)


class LoopOutputs:
    """A run's ``log`` and ``frames`` outputs, as the loop's sink."""

    def __init__(self, ctx: RunContext):
        self.events = ctx.output("log", LoopLogOutput)
        self.frames = ctx.output("frames", UnitFramesOutput)

    def log(self, event: LoopEvent) -> None:
        self.events.log(event)

    def add_frame(self, image: np.ndarray, channels: Sequence[str], tag: FrameTag) -> None:
        self.frames.add_frame(image, channels, tag)


def survey_map_of(ctx: RunContext) -> tuple[SurveyMap, SurveyOutput]:
    """The map of the survey read, and the survey itself."""
    survey = ctx.earlier("survey", SurveyOutput)
    overview = ctx.earlier("overview", PlacedImageOutput)
    # Compared by file, since each read may be loaded on its own from the same recording.
    if overview.recording.location != survey.recording.location:
        raise ParameterError("the overview read is not the survey's own: choose them together")
    found = survey.found()
    visits = tuple(
        Visit(tile.x_um, tile.y_um, taken.time_s)
        for tile, taken in zip(survey.tiles, survey.taken, strict=True)
    )
    return SurveyMap(found.focus, found.motion, Overview(*overview.latest()), visits), survey


def field_of(ctx: RunContext, survey: SurveyOutput, overscan: tuple[int, int]) -> FieldOfView:
    """The survey's field of view, scanned with the calibration read."""
    calibration = ctx.earlier("calibration", GalvoCalibrationOutput).measured()
    pixels = survey[0].shape[-1]
    return FieldOfView(
        pixels * survey.um_per_px,
        pixels,
        (calibration.um_per_volt_x, calibration.um_per_volt_y),
        ctx.inputs[DWELL_TIME_US],
        *overscan,
    )


def run_loop(
    ctx: RunContext,
    scope: Microscope,
    overscan: tuple[int, int],
    routines: tuple[RoutineSpec, ...],
) -> None:
    """Run ``routines`` on every unit drawn over the survey read."""
    survey_map, survey = survey_map_of(ctx)
    field = field_of(ctx, survey, overscan)
    units = ctx.inputs[UNITS]
    check_units(units, field)
    focus_check_s, final_step_um = ctx.inputs[FOCUS_CHECK_S], ctx.inputs[FOCUS_FINAL_STEP_UM]
    setup = LoopSetup(survey_map, field, focus_check_s, final_step_um, routines)
    Orchestrator(ctx, scope, setup, LoopOutputs(ctx), RunSteering(ctx)).run(units)
