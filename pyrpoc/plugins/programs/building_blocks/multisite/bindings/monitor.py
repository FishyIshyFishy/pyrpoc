"""The monitor routine bound to a run: its program inputs, and the routine a program hands the
loop."""

from __future__ import annotations

from pyrpoc.core.program import RunContext
from pyrpoc.plugins.program_io.inputs import (
    DURATION_MIN,
    DWELL_TIME_US,
    PERIOD_S,
    TREAT,
    TREATED_GROUP,
)

from ..interfaces.microscope import DigitalLine
from ..routines.monitor import Monitoring, Treatment, monitor
from ..stages.loop.routine import RoutineSpec

MONITOR_INPUTS = (PERIOD_S, DURATION_MIN, DWELL_TIME_US)
# Read again before each wait, so an edit takes effect at a unit's next scan.
MONITOR_LIVE = (PERIOD_S, DURATION_MIN)
TREATMENT_INPUTS = (TREAT, TREATED_GROUP)


def monitor_routine(ctx: RunContext, line: DigitalLine) -> RoutineSpec:
    """The monitor routine, following the run's live period and duration and treating on
    ``line`` if the run asks for it."""
    treatment = Treatment(ctx.inputs[TREATED_GROUP], line) if ctx.inputs[TREAT] else None

    def now() -> Monitoring:
        return Monitoring(ctx.current(PERIOD_S), ctx.current(DURATION_MIN))

    return RoutineSpec("monitor", lambda unit, history: monitor(unit, history, now, treatment))
