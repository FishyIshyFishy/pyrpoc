"""``longitudinal_monitor``'s routine: treat the unit once if it is in the treated group, then
scan it every period for the duration, recording its mean intensity each time."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyrpoc.core.program import RunContext
from pyrpoc.core.program_input import ParameterError
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.plugins.program_io.inputs import (
    DURATION_MIN,
    DWELL_TIME_US,
    FOCUS_CHECK_S,
    FOCUS_FINAL_STEP_UM,
    PERIOD_S,
    TREAT,
    TREATED_GROUP,
    UNITS,
    Unit,
)
from pyrpoc.plugins.program_io.outputs import (
    GalvoCalibrationOutput,
    LoopLogOutput,
    PlacedImageOutput,
    SurveyOutput,
    UnitFramesOutput,
)

from ..interfaces.field import FieldOfView
from ..interfaces.microscope import DigitalLine, Microscope
from ..stages.loop.orchestrator import LoopSetup, Orchestrator, check_units
from ..stages.loop.routine import Record, Routine, RoutineSpec, Scan, Wait, seen_from
from ..stages.survey.layout import Overview


@dataclass(frozen=True)
class Monitoring:
    """How often and for how long a unit is scanned; read again before each wait, so an edit
    takes effect at the unit's next scan."""

    period_s: float
    duration_min: float


@dataclass(frozen=True)
class Treatment:
    """Which group is treated at the start, on which line; None treats nobody."""

    group: str
    line: DigitalLine


def monitor(unit: Unit, now: Callable[[], Monitoring], treatment: Treatment | None) -> Routine:
    if treatment is not None and unit.group == treatment.group:
        yield Scan((treatment.line,))
    started: float | None = None
    while True:
        seen = seen_from((yield Scan(())))
        started = seen.time_s if started is None else started
        yield Record("intensity", seen.means())
        settings = now()
        if seen.time_s - started >= settings.duration_min * 60.0:
            return
        yield Wait(settings.period_s)


MONITOR_INPUTS = (PERIOD_S, DURATION_MIN, DWELL_TIME_US)
# Read again before each wait, so an edit takes effect at a unit's next scan.
MONITOR_LIVE = (PERIOD_S, DURATION_MIN)
TREATMENT_INPUTS = (TREAT, TREATED_GROUP)
FOCUS_INPUTS = (FOCUS_CHECK_S, FOCUS_FINAL_STEP_UM)
# What every loop fills: its log, and every frame of every unit.
LOOP_EMITS: dict[str, type[ProgramOutput]] = {"log": LoopLogOutput, "frames": UnitFramesOutput}

# What every loop program reads: the survey, its overview, and the calibration its scans use.
LOOP_READS: dict[str, type[ProgramOutput]] = {
    "survey": SurveyOutput,
    "overview": PlacedImageOutput,
    "calibration": GalvoCalibrationOutput,
}


def survey_of(ctx: RunContext) -> tuple[SurveyOutput, Overview]:
    """The survey the run was handed and its overview."""
    survey = ctx.earlier("survey", SurveyOutput)
    overview = ctx.earlier("overview", PlacedImageOutput)
    if overview.recording is not survey.recording:
        raise ParameterError("the overview read is not the survey's own: choose them together")
    planes, placement = overview.latest()
    return survey, Overview(planes, placement)


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


def treatment_of(ctx: RunContext, line: DigitalLine) -> Treatment | None:
    if not ctx.inputs[TREAT]:
        return None
    return Treatment(ctx.inputs[TREATED_GROUP], line)


def run_monitor(
    ctx: RunContext, scope: Microscope, overscan: tuple[int, int], line: DigitalLine
) -> None:
    """Monitor every unit drawn over the survey read, treating on ``line``."""
    survey, overview = survey_of(ctx)
    field = field_of(ctx, survey, overscan)
    units = ctx.inputs[UNITS]
    check_units(units, field)
    treatment = treatment_of(ctx, line)

    def now() -> Monitoring:
        return Monitoring(ctx.current(PERIOD_S), ctx.current(DURATION_MIN))

    spec = RoutineSpec("monitor", lambda unit: monitor(unit, now, treatment))
    focus_check_s, final_step_um = ctx.inputs[FOCUS_CHECK_S], ctx.inputs[FOCUS_FINAL_STEP_UM]
    setup = LoopSetup(
        survey, survey.found(), overview, field, focus_check_s, final_step_um, (spec,)
    )
    Orchestrator(ctx, scope, setup, units).run()
