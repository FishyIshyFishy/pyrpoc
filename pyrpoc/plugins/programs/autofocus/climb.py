"""The z search: a hill climb that halves its step at each overshoot and stops
when a halving no longer pays. Knows nothing of stages or scans; ``measure``
does, so the search can be tested against any curve."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..building_blocks.parameter_groups import FocusSearchGroup


@dataclass(frozen=True)
class ClimbStep:
    """One measurement: where, what it scored, and the step that got there
    (0 for the starting point)."""

    z_um: float
    metric: float
    step_um: float


@dataclass(frozen=True)
class ClimbResult:
    best_z_um: float
    best_metric: float
    # The metric was still rising at the edge of the search range, so the
    # peak may lie beyond it.
    at_edge: bool


@dataclass
class Climber:
    """The best point so far and how to probe from it."""

    probe: Callable[[float, float], float]
    low_um: float
    high_um: float
    best_z_um: float
    best_metric: float

    def climb_round(self, step_um: float, direction: float) -> tuple[float, bool]:
        """Step from the best point while the metric rises. If the very first
        probe falls, try the other side once, since the peak may lie there.
        Returns the direction last climbed and whether it ran into the range edge."""
        moved = False
        tried_other_side = False
        while True:
            target = min(max(self.best_z_um + direction * step_um, self.low_um), self.high_um)
            if target == self.best_z_um:
                return direction, True
            metric = self.probe(target, step_um)
            if metric > self.best_metric:
                self.best_z_um, self.best_metric = target, metric
                moved = True
            elif moved or tried_other_side:
                return direction, False
            else:
                tried_other_side = True
                direction = -direction


def relative_gain(before: float, after: float) -> float:
    """How much ``after`` improves on ``before``, as a fraction of it. Any gain
    from nothing is unbounded, so a dark start never reads as a plateau."""
    if before <= 0.0:
        return float("inf") if after > before else 0.0
    return (after - before) / before


def climb(
    measure: Callable[[float], float],
    report: Callable[[ClimbStep], None],
    start_z_um: float,
    search: FocusSearchGroup,
) -> ClimbResult:
    """Climb from ``start_z_um`` within its search range. ``measure`` scores a z;
    ``report`` is told of every measurement as it is made. The first round
    always refines once, so a start already near the peak still narrows in."""

    def probe(z_um: float, step_um: float) -> float:
        metric = measure(z_um)
        report(ClimbStep(z_um, metric, step_um))
        return metric

    climber = Climber(
        probe,
        start_z_um - search.search_range_um,
        start_z_um + search.search_range_um,
        start_z_um,
        probe(start_z_um, 0.0),
    )
    step_um, direction, first_round = search.initial_step_um, 1.0, True
    while True:
        before = climber.best_metric
        direction, at_edge = climber.climb_round(step_um, direction)
        if at_edge:
            return ClimbResult(climber.best_z_um, climber.best_metric, at_edge=True)
        gain = relative_gain(before, climber.best_metric)
        if not first_round and gain < search.plateau_pct / 100.0:
            return ClimbResult(climber.best_z_um, climber.best_metric, at_edge=False)
        first_round = False
        step_um, direction = step_um / 2.0, -direction
