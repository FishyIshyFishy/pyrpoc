"""The z search: a hill climb that steps while the metric rises and halves its
step when neither direction does, down to a final step. Knows nothing of stages
or scans; ``measure`` does, so the search can be tested against any curve."""

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


def climb(
    measure: Callable[[float], float],
    report: Callable[[ClimbStep], None],
    start_z_um: float,
    search: FocusSearchGroup,
) -> ClimbResult:
    """Climb from ``start_z_um`` within its search range. ``measure`` scores a z;
    ``report`` is told of every measurement as it is made."""
    low_um = start_z_um - search.search_range_um
    high_um = start_z_um + search.search_range_um

    def probe(z_um: float, step_um: float) -> float:
        metric = measure(z_um)
        report(ClimbStep(z_um, metric, step_um))
        return metric

    best_z_um, best_metric = start_z_um, probe(start_z_um, 0.0)
    step_um, direction = search.initial_step_um, 1.0
    while step_um >= search.final_step_um:
        # The side last climbed first: the peak is most likely just past it.
        for side in (direction, -direction):
            moved = False
            while True:
                target = min(max(best_z_um + side * step_um, low_um), high_um)
                metric = probe(target, step_um)
                if metric <= best_metric:
                    break
                best_z_um, best_metric, moved = target, metric, True
                if target in (low_um, high_um):
                    return ClimbResult(best_z_um, best_metric, at_edge=True)
            if moved:
                direction = side
                break
        step_um /= 2.0
    return ClimbResult(best_z_um, best_metric, at_edge=False)
