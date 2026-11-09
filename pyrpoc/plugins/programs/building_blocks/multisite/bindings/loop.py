"""The loop stage bound to a run: its program inputs, what it reads and fills, its scratch, and
the call a program makes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pyrpoc.core.program import RunContext
from pyrpoc.core.program_input import ProgramInput
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.plugins.program_io.inputs import (
    DWELL_TIME_US,
    FOCUS_CHECK_S,
    FOCUS_FINAL_STEP_UM,
    UNITS,
)
from pyrpoc.plugins.program_io.outputs import (
    FrameTag,
    GalvoCalibrationOutput,
    LoopEvent,
    LoopLogOutput,
    SurveyOutput,
    Unit,
    UnitFramesOutput,
    UnitRecord,
    UnitsOutput,
)

from ..interfaces.field import FieldOfView
from ..interfaces.microscope import Microscope
from ..interfaces.overview import Overview
from ..stages.loop.orchestrator import LoopSetup, Orchestrator
from ..stages.loop.resume import resumption
from ..stages.loop.routine import RoutineSpec
from ..stages.loop.survey_map import SurveyMap, Visit
from .scratch import Scratch

# Every loop program reads the units, live: they are what it runs on, however they change.
LOOP_UNITS: list[ProgramInput[Any]] = [UNITS]
LOOP_FOCUS_INPUTS = (FOCUS_CHECK_S, FOCUS_FINAL_STEP_UM)
# What every loop fills: its log, its units, and every frame of every unit.
LOOP_EMITS: dict[str, type[ProgramOutput]] = {
    "log": LoopLogOutput,
    "units": UnitsOutput,
    "frames": UnitFramesOutput,
}
# What every loop reads: the survey, and the calibration its scans use.
LOOP_READS: dict[str, type[ProgramOutput]] = {
    "survey": SurveyOutput,
    "calibration": GalvoCalibrationOutput,
}


class RunSteering:
    """The units a run is given now."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    def units(self) -> tuple[Unit, ...]:
        return self.ctx.current(UNITS)


class LoopOutputs:
    """A run's ``log``, ``units`` and ``frames`` outputs, and the loop's scratch, as the loop's
    sink. Each event goes to both the log and the scratch, and is the run's status."""

    def __init__(self, ctx: RunContext, scratch: Scratch):
        self.ctx = ctx
        self.events = ctx.output("log", LoopLogOutput)
        self.units = ctx.output("units", UnitsOutput)
        self.frames = ctx.output("frames", UnitFramesOutput)
        self.scratch = scratch

    def log(self, event: LoopEvent) -> None:
        self.scratch.append(event)
        self.events.log(event)
        self.ctx.status(event.text())

    def keep_units(self, records: tuple[UnitRecord, ...]) -> None:
        self.units.keep(records)

    def add_frame(self, image: np.ndarray, channels: Sequence[str], tag: FrameTag) -> None:
        self.frames.add_frame(image, channels, tag)


def survey_map_of(ctx: RunContext) -> tuple[SurveyMap, SurveyOutput]:
    """The map of the survey read, and the survey itself."""
    survey = ctx.earlier("survey", SurveyOutput)
    found = survey.found()
    visits = tuple(
        Visit(tile.x_um, tile.y_um, taken.time_s)
        for tile, taken in zip(survey.tiles, survey.taken, strict=True)
    )
    return SurveyMap(found.focus, found.motion, Overview(*survey.overview()), visits), survey


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


def scratch_beside(survey: SurveyOutput) -> Scratch:
    """The scratch in the survey's folder, which a loop reads from and is saved in."""
    folder: Path | None = survey.recording.folder
    if folder is None:
        raise ValueError("the survey read has no folder, so the loop has nowhere to keep work")
    return Scratch(folder)


def begin(ctx: RunContext, orchestrator: Orchestrator, outputs: LoopOutputs) -> None:
    """Carry on the loop the scratch holds, or start afresh on the units drawn."""
    scratch = outputs.scratch
    if not scratch.exists():
        note_recording(outputs)
        orchestrator.start(ctx.inputs[UNITS])
        return
    earlier = scratch.events()
    note_recording(outputs)
    # The log of the run that finishes the loop tells all of it, from its first event.
    outputs.events.log_all(earlier)
    orchestrator.resume(resumption(earlier))


def note_recording(outputs: LoopOutputs) -> None:
    """Name this run's recording in the scratch, so the folder check lets it be carried on."""
    root = outputs.events.root
    if root is not None:
        outputs.scratch.note_recording(root.name)


def run_loop(
    ctx: RunContext,
    scope: Microscope,
    overscan: tuple[int, int],
    routines: tuple[RoutineSpec, ...],
) -> None:
    """Run ``routines`` on every unit over the survey read, carrying on a loop cut short in the
    folder if there is one. The scratch goes once every routine has ended."""
    survey_map, survey = survey_map_of(ctx)
    field = field_of(ctx, survey, overscan)
    focus_check_s, final_step_um = ctx.inputs[FOCUS_CHECK_S], ctx.inputs[FOCUS_FINAL_STEP_UM]
    setup = LoopSetup(survey_map, field, focus_check_s, final_step_um, routines)
    outputs = LoopOutputs(ctx, scratch_beside(survey))
    orchestrator = Orchestrator(ctx, scope, setup, outputs, RunSteering(ctx))
    begin(ctx, orchestrator, outputs)
    orchestrator.run()
    outputs.scratch.discard()
