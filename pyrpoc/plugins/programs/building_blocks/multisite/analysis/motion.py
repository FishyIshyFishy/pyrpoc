"""How fast what was imaged moves: features in an early frame found again in a later frame of
the same place, and the distance each moved over the time between."""

from __future__ import annotations

import numpy as np

from .features import Box, clear_features
from .shifts import locate_near

EDGE_MARGIN_PX = 4
# How far a feature is looked for, as a fraction of the field.
SEARCH_FRACTION = 0.2


def feature_speeds(
    before: np.ndarray, after: np.ndarray, minutes: float, um_per_px: float
) -> list[float]:
    """The speed, in µm per minute, of each clear feature of plane ``before`` found again in
    plane ``after``, ``minutes`` later."""
    height, width = before.shape
    radius = SEARCH_FRACTION * max(height, width)
    speeds = []
    for feature in clear_features(before, Box(0, 0, height, width)):
        region = feature.grown(EDGE_MARGIN_PX, (height, width))
        corner = (float(region.left), float(region.top))
        found = locate_near(region.crop(before), after, corner, radius)
        if found is not None:
            moved_px = float(np.hypot(found[0] - corner[0], found[1] - corner[1]))
            speeds.append(moved_px * um_per_px / minutes)
    return speeds
