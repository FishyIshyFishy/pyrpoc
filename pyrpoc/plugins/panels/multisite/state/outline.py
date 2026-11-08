"""A drawn outline: thinned to the points that shape it, and whether it is worth keeping."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pyrpoc.plugins.program_io.outputs import Outline

# The smallest share of a field a drawn unit may span, so a stray drag draws nothing.
SMALLEST = 0.03
# The largest share of a field a unit may span, leaving the loop room to track it.
LARGEST = 0.8


def thinned(points: Sequence[tuple[float, float]], tolerance: float) -> Outline:
    """``points`` without those within ``tolerance`` of the line their neighbours make."""
    if len(points) < 3:
        return tuple(points)
    array = np.asarray(points, dtype=float)
    keep = np.zeros(len(array), dtype=bool)
    keep[0] = keep[-1] = True
    spans = [(0, len(array) - 1)]
    while spans:
        start, end = spans.pop()
        if end - start < 2:
            continue
        far, distance = farthest(array, start, end)
        if distance > tolerance:
            keep[far] = True
            spans += [(start, far), (far, end)]
    return tuple((float(x), float(y)) for x, y in array[keep])


def farthest(array: np.ndarray, start: int, end: int) -> tuple[int, float]:
    """The point between ``start`` and ``end`` farthest from the line through them."""
    a, b = array[start], array[end]
    inner = array[start + 1 : end]
    line = b - a
    length = float(np.hypot(line[0], line[1]))
    offsets = inner - a
    if length == 0.0:
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
    else:
        distances = np.abs(line[0] * offsets[:, 1] - line[1] * offsets[:, 0]) / length
    index = int(np.argmax(distances))
    return start + 1 + index, float(distances[index])


def middle(outline: Outline) -> tuple[float, float]:
    """The centre of the outline's bounds, in µm: where a marker for it goes."""
    points = np.asarray(outline)
    low, high = points.min(axis=0), points.max(axis=0)
    return float((low[0] + high[0]) / 2), float((low[1] + high[1]) / 2)


def span_um(outline: Outline) -> float:
    """The outline's longer side, in µm."""
    points = np.asarray(outline)
    return float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1])))


def too_small(outline: Outline, field_um: float) -> bool:
    return len(outline) < 3 or span_um(outline) < SMALLEST * field_um


def fits(outline: Outline, field_um: float) -> bool:
    """Whether a unit this shape leaves the loop room to track it within one field."""
    return span_um(outline) <= LARGEST * field_um
