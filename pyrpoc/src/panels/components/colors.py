"""The channel colour palette shared by the overlay and spectrum panels."""

from __future__ import annotations

PALETTE = [
    (255, 80, 80),
    (80, 220, 120),
    (70, 150, 255),
    (255, 200, 70),
    (190, 110, 255),
    (70, 230, 230),
    (255, 120, 210),
    (180, 180, 180),
]


def color_for_index(index: int) -> tuple[int, int, int]:
    """A channel's colour, stable across panels so it keeps its identity."""
    return PALETTE[index % len(PALETTE)]
