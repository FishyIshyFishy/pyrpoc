"""Pin both samples-per-pixel formulas.

The raster path truncates with a floor of 1; the FLIM path rounds with a floor
of 2, because the counter that derives the pixel clock needs one high tick and
one low tick. Neither was covered by a test before, so unifying them while
moving the code would have been a silent behaviour change on the instrument.
"""

from __future__ import annotations

import pytest

from pyrpoc.operations.raster import pixel_samples as raster_pixel_samples
from pyrpoc.operations.tagger import pixel_samples as flim_pixel_samples


@pytest.mark.parametrize(
    "dwell_us,rate_hz,expected",
    [
        (2.0, 100_000.0, 0),      # 0.2 -> truncates to 0, floored to 1
        (10.0, 100_000.0, 1),     # 1.0
        (2.0, 1_000_000.0, 2),    # 2.0
        (2.9, 1_000_000.0, 2),    # 2.9 truncates to 2, NOT rounded to 3
        (0.1, 1_000_000.0, 0),    # 0.1 -> floored to 1
    ],
)
def test_raster_truncates_with_floor_one(dwell_us, rate_hz, expected):
    assert raster_pixel_samples(dwell_us, rate_hz) == max(1, expected)


@pytest.mark.parametrize(
    "dwell_us,rate_hz,expected",
    [
        (2.0, 100_000.0, 2),      # 0.2 rounds to 0, floored to 2
        (2.0, 1_000_000.0, 2),    # 2.0
        (2.9, 1_000_000.0, 3),    # 2.9 ROUNDS to 3, unlike the raster path
        (2.4, 1_000_000.0, 2),
        (10.0, 1_000_000.0, 10),
    ],
)
def test_flim_rounds_with_floor_two(dwell_us, rate_hz, expected):
    assert flim_pixel_samples(dwell_us, rate_hz) == expected


def test_the_two_formulas_genuinely_differ():
    """If this ever passes as equal, someone unified them. Do not."""
    assert raster_pixel_samples(2.9, 1_000_000.0) == 2
    assert flim_pixel_samples(2.9, 1_000_000.0) == 3
    assert raster_pixel_samples(0.5, 1_000_000.0) == 1
    assert flim_pixel_samples(0.5, 1_000_000.0) == 2
