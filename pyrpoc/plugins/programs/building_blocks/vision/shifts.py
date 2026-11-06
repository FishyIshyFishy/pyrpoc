"""Where a template sits in a plane, to a fraction of a pixel."""

from __future__ import annotations

import cv2
import numpy as np

# Below this normalised correlation the template is taken as not found.
MATCH_FLOOR = 0.5


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


def locate_near(
    template: np.ndarray, plane: np.ndarray, expected: tuple[float, float], radius: float
) -> tuple[float, float] | None:
    """The (x, y) of the template's top-left corner in ``plane``, searched for within
    ``radius`` pixels of ``expected``, or None when it is not found there."""
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
    scores = cv2.matchTemplate(window, template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    x, y, strength = peak(scores)
    if strength < MATCH_FLOOR:
        return None
    return left + x, top + y
