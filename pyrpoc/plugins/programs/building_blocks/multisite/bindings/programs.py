"""``CalibrationProgram``, ``SurveyProgram`` and ``LoopProgram``: the three kinds of multisite
program. Each declares what every program of its kind reads and fills, what it needs before it
starts, and runs its stage; a program subclasses one and gives its microscope and its own inputs."""

from __future__ import annotations

from typing import Any

from pyrpoc.core.program import PreflightContext, Program, RunContext
from pyrpoc.core.program_input import ProgramInput
from pyrpoc.core.runner import Runner
from pyrpoc.plugins.program_io.inputs import SCAN_GEOMETRY_INPUTS, ScanGeometry

from ...runners import Single
from ..interfaces.microscope import Microscope
from ..stages.loop.routine import RoutineSpec
from .calibration import CALIBRATION_EMITS, run_calibration
from .loop import LOOP_EMITS, LOOP_FOCUS_INPUTS, LOOP_READS, LOOP_UNITS, run_loop
from .preflight import calibration_measured, clean_folder, starting_units, survey_finished
from .survey import (
    REGION_INPUTS,
    SURVEY_EMITS,
    SURVEY_FOCUS_INPUTS,
    SURVEY_READS,
    TILE_INPUTS,
    run_survey,
)

# Each step of the pipeline runs once, in a folder holding only the steps before it.
RUNNERS: list[Runner] = [Single()]


class CalibrationProgram(Program):
    """Measures the scanner's µm per volt, the pipeline's first step."""

    inputs: list[ProgramInput[Any]] = [*SCAN_GEOMETRY_INPUTS]
    cards = {"Scan": SCAN_GEOMETRY_INPUTS}
    emits = CALIBRATION_EMITS
    runners = RUNNERS

    def preflight(self, ctx: PreflightContext) -> list[str]:
        return clean_folder(ctx, [], carries_on=False)

    def microscope(self, ctx: RunContext) -> Microscope:
        """The instrument the run claimed."""
        raise NotImplementedError

    def run(self, ctx: RunContext) -> None:
        run_calibration(ctx, self.microscope(ctx), ScanGeometry.from_inputs(ctx.inputs))


class SurveyProgram(Program):
    """Maps a region from the stage's corner, with the calibration read."""

    # The calibration has no row: the engine sets it from the save folder.
    inputs: list[ProgramInput[Any]] = [
        *REGION_INPUTS,
        *TILE_INPUTS,
        *SURVEY_FOCUS_INPUTS,
        *SURVEY_READS,
    ]
    cards = {"Region": REGION_INPUTS, "Tiles": TILE_INPUTS, "Focus Search": SURVEY_FOCUS_INPUTS}
    emits = SURVEY_EMITS
    runners = RUNNERS

    def preflight(self, ctx: PreflightContext) -> list[str]:
        return [*clean_folder(ctx, SURVEY_READS, carries_on=False), *calibration_measured(ctx)]

    def microscope(self, ctx: RunContext) -> Microscope:
        """The instrument the run claimed."""
        raise NotImplementedError

    def overscan(self, ctx: RunContext) -> tuple[int, int]:
        """Extra pixels scanned left and right of each line, for the mirrors to turn around."""
        raise NotImplementedError

    def run(self, ctx: RunContext) -> None:
        run_survey(ctx, self.microscope(ctx), self.overscan(ctx))


class LoopProgram(Program):
    """Runs its routines on the units drawn over the survey read, taking edits to them live,
    and carries on a loop cut short in its folder."""

    # The units have no row: they are drawn over the survey. Nor do the survey and calibration:
    # the engine sets them from the save folder.
    inputs: list[ProgramInput[Any]] = [*LOOP_UNITS, *LOOP_FOCUS_INPUTS, *LOOP_READS]
    cards = {"Focus": LOOP_FOCUS_INPUTS}
    live: list[ProgramInput[Any]] = [*LOOP_UNITS]
    emits = LOOP_EMITS
    runners = RUNNERS

    def preflight(self, ctx: PreflightContext) -> list[str]:
        return [
            *clean_folder(ctx, LOOP_READS, carries_on=True),
            *calibration_measured(ctx),
            *survey_finished(ctx),
            *starting_units(ctx),
        ]

    def microscope(self, ctx: RunContext) -> Microscope:
        """The instrument the run claimed."""
        raise NotImplementedError

    def overscan(self, ctx: RunContext) -> tuple[int, int]:
        """Extra pixels scanned left and right of each line, for the mirrors to turn around."""
        raise NotImplementedError

    def routines(self, ctx: RunContext) -> tuple[RoutineSpec, ...]:
        """What the loop does to each unit."""
        raise NotImplementedError

    def run(self, ctx: RunContext) -> None:
        run_loop(ctx, self.microscope(ctx), self.overscan(ctx), self.routines(ctx))
