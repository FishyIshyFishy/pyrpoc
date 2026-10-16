"""The z search. A coarse search rings outward from the start until it has seen
a significant peak fall away on both sides, so its cost grows with the distance
to focus, not with the range. A fine sweep across that peak, always moving up so
the drive's backlash cancels, is then fitted for the focus. Knows nothing of
stages or scans; ``measure`` and ``move`` do, so the search can be tested
against any curve."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from pyrpoc.structs.plugins.params import ParameterError

from ..building_blocks.parameter_groups import FocusSearchGroup


@dataclass(frozen=True)
class FocusStep:
    """One measurement: where, what it scored, and which phase took it
    ("bracket" or "refine")."""

    z_um: float
    metric: float
    phase: str


@dataclass(frozen=True)
class FocusResult:
    best_z_um: float
    best_metric: float
    # The peak was still rising at the edge of the search range, so the
    # focus may lie beyond it.
    at_edge: bool
    # Whether best_z_um is a fitted vertex rather than the sharpest sample.
    fitted: bool


Probe = Callable[[float, str], FocusStep]


def outward_targets(
    start_um: float, low_um: float, high_um: float, step_um: float
) -> Iterator[float]:
    """Alternately above and below the start, one step further each time, each
    side ending on its range edge."""
    up = down = True
    k = 1
    while up or down:
        if up:
            z = start_um + k * step_um
            up = z < high_um
            yield min(z, high_um)
        if down:
            z = start_um - k * step_um
            down = z > low_um
            yield max(z, low_um)
        k += 1


def significant_peak(steps: list[FocusStep], min_contrast: float) -> FocusStep | None:
    """The sharpest step, if it stands out from the dullest by ``min_contrast``.
    Noise alone has no contrast, so a field with no focus finds no peak."""
    peak = max(steps, key=lambda step: step.metric)
    floor = min(step.metric for step in steps)
    if peak.metric > floor and peak.metric >= min_contrast * floor:
        return peak
    return None


def is_bracketed(steps: list[FocusStep], peak: FocusStep) -> bool:
    """Whether the metric has fallen below half the peak's height on both sides
    of it, so the peak itself was seen and not just its shoulder."""
    floor = min(step.metric for step in steps)
    half = floor + (peak.metric - floor) / 2.0
    below = [step.z_um for step in steps if step.metric < half]
    return any(z < peak.z_um for z in below) and any(z > peak.z_um for z in below)


def bracket(
    probe: Probe, start_um: float, low_um: float, high_um: float, search: FocusSearchGroup
) -> FocusStep | None:
    """Ring outward until a significant peak is bracketed, or the range runs
    out. Returns the coarse peak, or None if nothing stood out."""
    steps = [probe(start_um, "bracket")]
    for z_um in outward_targets(start_um, low_um, high_um, search.coarse_step_um):
        steps.append(probe(z_um, "bracket"))
        peak = significant_peak(steps, search.min_contrast)
        if peak is not None and is_bracketed(steps, peak):
            return peak
    return significant_peak(steps, search.min_contrast)


def approach(move: Callable[[float], None], z_um: float, low_um: float, distance_um: float) -> None:
    """Arrive at ``z_um`` moving up, as the fine sweep did, so backlash lands
    the drive where the sweep measured."""
    move(max(z_um - distance_um, low_um))
    move(z_um)


def refine(
    probe: Probe,
    move: Callable[[float], None],
    center_um: float,
    low_um: float,
    high_um: float,
    search: FocusSearchGroup,
) -> list[FocusStep]:
    """Sweep up across one coarse step either side of ``center_um``."""
    first_um = max(center_um - search.coarse_step_um, low_um)
    last_um = min(center_um + search.coarse_step_um, high_um)
    approach(move, first_um, low_um, search.coarse_step_um)
    count = int((last_um - first_um) / search.fine_step_um + 1e-9)
    return [probe(first_um + i * search.fine_step_um, "refine") for i in range(count + 1)]


def fit_peak(steps: list[FocusStep]) -> float | None:
    """The vertex of a Gaussian (a parabola in log metric) fitted to the points
    above half height around the sharpest one. None when the points do not
    make a peak inside their span."""
    z = np.array([step.z_um for step in steps])
    metric = np.array([step.metric for step in steps])
    if np.any(metric <= 0.0):
        return None
    top = int(np.argmax(metric))
    half = (metric.min() + metric[top]) / 2.0
    left, right = top, top
    while left > 0 and metric[left - 1] > half:
        left -= 1
    while right < len(metric) - 1 and metric[right + 1] > half:
        right += 1
    # A peak narrower than the fine step still gets its two neighbours.
    left, right = min(left, max(top - 1, 0)), max(right, min(top + 1, len(metric) - 1))
    if right - left < 2:
        return None
    a, b, _ = np.polyfit(z[left : right + 1] - z[top], np.log(metric[left : right + 1]), 2)
    if a >= 0.0:
        return None
    vertex = float(z[top] - b / (2.0 * a))
    return vertex if z[left] <= vertex <= z[right] else None


def search_focus(
    measure: Callable[[float], float],
    move: Callable[[float], None],
    report: Callable[[FocusStep], None],
    start_z_um: float,
    search: FocusSearchGroup,
) -> FocusResult | None:
    """Search within the range around ``start_z_um`` and leave the drive at the
    focus, or back at the start if none was found (None). ``measure`` moves to
    and scores a z; ``move`` only moves; ``report`` is told of every
    measurement as it is made."""
    if search.fine_step_um >= search.coarse_step_um:
        raise ParameterError("the fine step must be smaller than the coarse step")
    low_um = start_z_um - search.search_range_um
    high_um = start_z_um + search.search_range_um

    def probe(z_um: float, phase: str) -> FocusStep:
        step = FocusStep(z_um, measure(z_um), phase)
        report(step)
        return step

    peak = bracket(probe, start_z_um, low_um, high_um, search)
    if peak is None:
        approach(move, start_z_um, low_um, search.coarse_step_um)
        return None
    fine = refine(probe, move, peak.z_um, low_um, high_um, search)
    sharpest = max(fine, key=lambda step: step.metric)
    vertex = fit_peak(fine)
    best_z_um = sharpest.z_um if vertex is None else vertex
    approach(move, best_z_um, low_um, search.coarse_step_um)
    return FocusResult(
        best_z_um,
        sharpest.metric,
        at_edge=peak.z_um in (low_um, high_um),
        fitted=vertex is not None,
    )
