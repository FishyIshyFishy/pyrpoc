"""Keeping focus through a loop: the survey's focus plane plus an offset that drifts, followed
through the periodic checks by a Kalman filter."""

# The offset drifts at a rate that itself changes slowly: the nearly-constant-velocity model
# (Bar-Shalom, Li and Kirubarajan 2001, section 6.2). A microscope's thermal drift runs steadily
# over minutes and settles over hours, so the filter predicts the offset between checks from the
# rate, and knows how far focus may have gone since the last check, which sizes the next search.
# A check falls due at the user's interval, or sooner while the filter cannot yet predict focus to
# the final step, as at the start, before it has seen the drift's rate.
#
# The filter starts from the survey: no offset from its plane, as it stood when the last tile was
# taken, and the drift the survey measured, or none known if it measured none.
#
# A check's error is the scatter of focus about the survey's plane: it holds both the search's own
# error and how far the dish departs from a plane between where a check is made and a unit.

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np

from pyrpoc.plugins.program_io.inputs import FocusSearch
from pyrpoc.plugins.program_io.outputs import FocusDrift, FocusPlane

from ....focus.climb import ClimbResult, find_focus

# How fast focus may be drifting before any check says (one deviation), and how long the rate
# takes to change by as much: a dish settling after it is mounted drifts at up to about this.
RATE_DEVIATION_UM_PER_MIN = 0.3
RATE_CHANGE_MIN = 30.0
# How many deviations either side of the prediction a check searches.
SEARCH_DEVIATIONS = 3.0
# A check this unlikely under the filter (chi-squared with one degree, 1 in 1000) says focus
# jumped, as when the dish is knocked, and the filter starts again from it.
JUMP_SCORE = 10.83


class FocusKeeper:
    """Where focus is: the survey's plane, never refitted, plus an offset from it and the
    offset's rate of drift, estimated with their uncertainty from every check so far."""

    def __init__(
        self,
        plane: FocusPlane,
        drift: FocusDrift | None,
        plane_s: float,
        every_s: float,
        full: FocusSearch,
    ):
        self.plane = plane
        self.every_s = every_s
        self.full = full
        self.noise = plane.residual_um**2
        # The rate's random walk, as the spectral density of its change per second.
        rate = RATE_DEVIATION_UM_PER_MIN / 60.0
        self.rate_change = rate**2 / (RATE_CHANGE_MIN * 60.0)
        self.state, self.covariance = self.fresh(0.0)
        if drift is not None:
            self.state[1] = drift.um_per_min / 60.0
            self.covariance[1, 1] = (drift.error_um_per_min / 60.0) ** 2
        self.at_s = plane_s
        self.checked_s = -math.inf
        # Whether the last check found no focus in range, so the next waits the whole interval.
        self.lost = False

    def fresh(self, offset_um: float) -> tuple[np.ndarray, np.ndarray]:
        """A state at ``offset_um`` known as well as one check knows it, and its rate not at
        all."""
        rate = RATE_DEVIATION_UM_PER_MIN / 60.0
        return np.array([offset_um, 0.0]), np.diag([self.noise, rate**2])

    def predicted(self, now_s: float) -> tuple[np.ndarray, np.ndarray]:
        """The offset and its rate at ``now_s``, and their covariance."""
        elapsed = max(0.0, now_s - self.at_s)
        transition = np.array([[1.0, elapsed], [0.0, 1.0]])
        drift = self.rate_change * np.array(
            [[elapsed**3 / 3.0, elapsed**2 / 2.0], [elapsed**2 / 2.0, elapsed]]
        )
        state = transition @ self.state
        covariance = transition @ self.covariance @ transition.T + drift
        return state, covariance

    def z_at(self, x_um: float, y_um: float, now_s: float) -> float:
        return self.plane.z_at(x_um, y_um) + float(self.predicted(now_s)[0][0])

    def due(self, now_s: float) -> bool:
        """Whether a check is due: the interval has passed since the last, or the filter is no
        longer sure of focus to the final step."""
        if now_s - self.checked_s >= self.every_s:
            return True
        deviation = math.sqrt(float(self.predicted(now_s)[1][0, 0]))
        return not self.lost and deviation > self.full.final_step_um

    def search(self, now_s: float) -> FocusSearch:
        """As far either side of the prediction as focus may be at ``now_s``, at least a few
        final steps and at most the whole range."""
        deviation = math.sqrt(float(self.predicted(now_s)[1][0, 0]) + self.noise)
        span = SEARCH_DEVIATIONS * deviation
        span = min(max(span, 4.0 * self.full.final_step_um), self.full.search_range_um)
        step = min(self.full.initial_step_um, span / 2.0)
        return FocusSearch(span, step, self.full.final_step_um)

    def find(
        self, measure: Callable[[float], float], x_um: float, y_um: float, now_s: float
    ) -> ClimbResult:
        """The sharpest z at (``x_um``, ``y_um``), scoring each z with ``measure``: searched for
        as far as focus may have gone, and over the whole range if that ends at its edge."""
        guess = self.z_at(x_um, y_um, now_s)
        search = self.search(now_s)
        found = find_focus(measure, lambda _step: None, guess, search)
        if found.at_edge and search != self.full:
            found = find_focus(measure, lambda _step: None, guess, self.full)
        return found

    def take(self, now_s: float, offset_um: float) -> None:
        """Update the offset with a check at ``now_s`` that found focus ``offset_um`` off the
        plane."""
        state, covariance = self.predicted(now_s)
        innovation = offset_um - state[0]
        spread = covariance[0, 0] + self.noise
        if innovation**2 / spread > JUMP_SCORE:
            state, covariance = self.fresh(offset_um)
        else:
            gain = covariance[:, 0] / spread
            state = state + gain * innovation
            covariance = covariance - np.outer(gain, covariance[0, :])
        self.state, self.covariance = state, covariance
        self.at_s = self.checked_s = now_s
        self.lost = False

    def missed(self, now_s: float) -> None:
        """A check at ``now_s`` found no focus in its whole range; the offset stays as
        predicted."""
        self.checked_s = now_s
        self.lost = True

    def replay(self, checks: Iterable[tuple[float, float]]) -> None:
        """Take each (time, offset) check of a loop cut short, in order."""
        for time_s, offset_um in checks:
            self.take(time_s, offset_um)
