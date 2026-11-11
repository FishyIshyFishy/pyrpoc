"""Where a focus check is made: in the nearest field beside a unit that the survey saw content
in and that keeps clear of every unit."""

# A check scans its field a dozen times or so. Made on a unit, that light lands on whichever
# unit is being scanned when a check falls due, which with a steady period is the same unit every
# time, and it bleaches that unit apart from the rest. Beside the units, focus is found on the same
# dish a field or two away, and the survey's plane carries it over to them.

from __future__ import annotations

import math
from collections.abc import Sequence

from ...analysis.content import content_channels
from ...interfaces.overview import Overview

# (left, top, right, bottom) in µm on the dish.
Bounds = tuple[float, float, float, float]

# Field centres around a unit, in fields, out to two fields away, nearest first.
OFFSETS = sorted(
    ((dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)),
    key=lambda offset: math.hypot(*offset),
)


def overlaps(a: Bounds, b: Bounds) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def surveyed_content(overview: Overview, bounds: Bounds) -> bool:
    """Whether ``bounds`` lies within the overview and the overview shows content there."""
    placement = overview.placement
    scale = placement.um_per_px
    left = (bounds[0] - placement.left_um) / scale
    top = (bounds[1] - placement.top_um) / scale
    right = (bounds[2] - placement.left_um) / scale
    bottom = (bounds[3] - placement.top_um) / scale
    height, width = overview.planes.shape[-2:]
    if left < 0 or top < 0 or right > width or bottom > height:
        return False
    crop = overview.planes[:, int(top) : int(bottom), int(left) : int(right)]
    return bool(content_channels(crop))


def focus_site(
    near: tuple[float, float], avoid: Sequence[Bounds], overview: Overview, field_um: float
) -> tuple[float, float] | None:
    """The centre of the nearest field to ``near``, whole fields away, that holds content in
    the survey and overlaps none of ``avoid``; None if none within two fields does."""
    half = field_um / 2.0
    for dx, dy in OFFSETS:
        x, y = near[0] + dx * field_um, near[1] + dy * field_um
        bounds = (x - half, y - half, x + half, y + half)
        if any(overlaps(bounds, other) for other in avoid):
            continue
        if surveyed_content(overview, bounds):
            return x, y
    return None
