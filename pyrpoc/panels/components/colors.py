"""The channel colour palette shared by overlay and spectrum.

Was defined twice, byte for byte, with a comment on the second copy promising
it would be kept identical to the first. One definition instead of a promise.
"""

from __future__ import annotations


def color_for_index(index: int) -> tuple[int, int, int]:
    """A channel's colour, stable across panels so it keeps its identity."""
    palette = [
        (255, 80, 80),
        (80, 220, 120),
        (70, 150, 255),
        (255, 200, 70),
        (190, 110, 255),
        (70, 230, 230),
        (255, 120, 210),
        (180, 180, 180),
    ]
    return palette[index % len(palette)]
