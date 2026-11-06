"""Site Curves names measurements plainly and averages each group by time."""

from __future__ import annotations

import numpy as np

from pyrpoc.plugins.panels.curves.panel import group_means, measurement_labels


def test_a_measurement_is_named_by_itself_unless_two_routines_share_the_name() -> None:
    assert measurement_labels([("monitor", "intensity"), ("monitor", "area")]) == [
        "intensity",
        "area",
    ]
    assert measurement_labels([("monitor", "intensity"), ("treat", "intensity")]) == [
        "intensity (monitor)",
        "intensity (treat)",
    ]


def test_group_means_average_sites_within_each_period() -> None:
    first = ((0.0, (1.0,)), (60.0, (3.0,)))
    second = ((0.0, (3.0,)), (60.0, (5.0,)))
    minutes, means = group_means([first, second], 0)
    assert np.allclose(minutes, [0.0, 1.0])
    assert np.allclose(means, [2.0, 4.0])
