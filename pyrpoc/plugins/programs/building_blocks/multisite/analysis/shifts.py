"""Where a template sits in a plane, to a fraction of a pixel."""

# A match also says whether it stands clear of the next best place, outside its own peak, as in
# the peak-ratio test of correlation velocimetry (Keane and Adrian 1992). A template that matches
# two places nearly equally has been found at neither, so a caller that takes each match as a
# measurement of one thing must ask for a distinct one.

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Below this normalised correlation the template is taken as not found.
MATCH_FLOOR = 0.5
# How many times the next best place's correlation the best must reach.
PEAK_RATIO = 1.5


def parabola_peak(left: float, centre: float, right: float) -> float:
    """The offset from the centre sample of the peak of the parabola through three samples."""
    curvature = left - 2.0 * centre + right
    return 0.0 if curvature >= 0 else 0.5 * (left - right) / curvature


def peak(scores: np.ndarray) -> tuple[float, float, float]:
    """The sub-pixel (x, y) of the best score, and the score."""
    _, strength, _, (x, y) = cv2.minMaxLoc(scores)
    height, width = scores.shape
    dx = parabola_peak(*scores[y, x - 1 : x + 2]) if 0 < x < width - 1 else 0.0
    dy = parabola_peak(*scores[y - 1 : y + 2, x]) if 0 < y < height - 1 else 0.0
    return float(x + dx), float(y + dy), float(strength)


@dataclass(frozen=True)
class Match:
    """Where a template's top-left corner sits in a plane, in pixels (x, y), and whether
    nothing else searched matched nearly as well."""

    x: float
    y: float
    distinct: bool


def distinct(scores: np.ndarray, at: tuple[int, int], strength: float) -> bool:
    """Whether the best score, ``strength`` at whole pixel (x, y) ``at``, reaches
    ``PEAK_RATIO`` times the best outside its own peak: the scores joined to it above half
    of it."""
    high = (scores >= strength / 2.0).astype(np.uint8)
    _, labels = cv2.connectedComponents(high, connectivity=8)
    others = scores[labels != labels[at[1], at[0]]]
    if others.size == 0:
        return True
    second = float(others.max())
    return second <= 0.0 or strength >= PEAK_RATIO * second


def locate_near(
    template: np.ndarray,
    mask: np.ndarray,
    plane: np.ndarray,
    expected: tuple[float, float],
    radius: float,
) -> Match | None:
    """Where the template's top-left corner is in ``plane``, matched on the template's ``mask``
    pixels and searched for within ``radius`` pixels of (x, y) ``expected``, or None when it is
    not found there."""
    # A small window keeps a similar-looking neighbour from being taken for the template.
    t_height, t_width = template.shape
    height, width = plane.shape
    left = max(0, int(np.floor(expected[0] - radius)))
    top = max(0, int(np.floor(expected[1] - radius)))
    right = min(width, int(np.ceil(expected[0] + radius)) + t_width + 1)
    bottom = min(height, int(np.ceil(expected[1] + radius)) + t_height + 1)
    if right - left <= t_width or bottom - top <= t_height:
        return None
    window = plane[top:bottom, left:right].astype(np.float32)
    scores = cv2.matchTemplate(
        window, template.astype(np.float32), cv2.TM_CCOEFF_NORMED, mask=mask.astype(np.uint8)
    )
    # Where the window is flat under the mask the correlation is undefined: no match there.
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, strength, _, (best_x, best_y) = cv2.minMaxLoc(scores)
    if strength < MATCH_FLOOR:
        return None
    x, y, _ = peak(scores)
    return Match(left + x, top + y, distinct(scores, (best_x, best_y), strength))
