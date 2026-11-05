"""``longitudinal_monitor``'s routine: treat the unit once if it is in the treated group, then
scan it every period for the duration, recording its mean intensity each time."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyrpoc.core.program import EarlierRun, RunContext
from pyrpoc.core.program_input import ParameterError
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
from pyrpoc.plugins.program_io.outputs import Mosaic, Overview, SurveyResult

from ..field import Field
from ..microscope import DigitalLine, Microscope
from .orchestrator import LoopSetup, Orchestrator, check_units
from .steps import Record, Routine, RoutineSpec, Scan, Wait, seen_from


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


def survey_of(ctx: RunContext, run: EarlierRun) -> tuple[SurveyResult, Overview]:
    """The survey the run was handed and its overview, read from a recording on disk."""
    outputs = ctx.earlier(run)
    survey, mosaic = outputs["survey"].value, outputs["mosaic"].value
    if not isinstance(survey, SurveyResult) or not isinstance(mosaic, Mosaic):
        raise ParameterError("the Survey in this folder holds no result")
    if mosaic.overview is None:
        raise ParameterError("the Survey in this folder was not stitched")
    return survey, mosaic.overview


def treatment_of(ctx: RunContext, line: DigitalLine) -> Treatment | None:
    if not ctx.inputs[TREAT]:
        return None
    return Treatment(ctx.inputs[TREATED_GROUP], line)


def run_monitor(
    ctx: RunContext,
    scope: Microscope,
    run: EarlierRun,
    overscan: tuple[int, int],
    line: DigitalLine,
) -> None:
    """Monitor every unit drawn over the survey ``run``, treating on ``line``."""
    survey, overview = survey_of(ctx, run)
    field = Field(
        survey.field_um,
        survey.field_pixels,
        (survey.um_per_volt_x, survey.um_per_volt_y),
        ctx.inputs[DWELL_TIME_US],
        *overscan,
    )
    units = ctx.inputs[UNITS]
    check_units(units, field)
    treatment = treatment_of(ctx, line)

    def now() -> Monitoring:
        return Monitoring(ctx.current(PERIOD_S), ctx.current(DURATION_MIN))

    spec = RoutineSpec("monitor", lambda unit: monitor(unit, now, treatment))
    setup = LoopSetup(
        survey, overview, field, ctx.inputs[FOCUS_CHECK_S], ctx.inputs[FOCUS_FINAL_STEP_UM], (spec,)
    )
    Orchestrator(ctx, scope, setup, units).run()
