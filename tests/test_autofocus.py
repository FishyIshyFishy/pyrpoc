"""Autofocus without a rig: the z search, the region scan's geometry, and the
focus metric, each checked against a case whose answer is known."""

from __future__ import annotations

import math
from collections.abc import Callable

import cv2
import numpy as np
import pytest

from pyrpoc.plugins.programs.autofocus.climb import ClimbResult, ClimbStep, climb
from pyrpoc.plugins.programs.autofocus.roi_scan import (
    Box,
    metric_weights,
    place_rows,
    roi_region,
    roi_waveform,
    row_spans,
)
from pyrpoc.plugins.programs.autofocus.tenengrad import tenengrad
from pyrpoc.plugins.programs.building_blocks.functions.galvo_raster import raster_waveform
from pyrpoc.plugins.programs.building_blocks.parameter_groups import (
    FocusSearchGroup,
    Mask,
    MaskRef,
    RoiGroup,
    ScanGroup,
)
from pyrpoc.structs.plugins.params import ParameterError

SEARCH = FocusSearchGroup(search_range_um=50.0, initial_step_um=5.0, plateau_pct=1.0)


def small_scan() -> ScanGroup:
    return ScanGroup(
        x_pixels=12,
        y_pixels=10,
        extra_left=2,
        extra_right=1,
        fast_axis_offset=0.1,
        fast_axis_amplitude=0.8,
        slow_axis_offset=-0.2,
        slow_axis_amplitude=0.5,
    )


def u_shape() -> np.ndarray:
    """Two prongs on rows 2-3 joined by a base on rows 4-5: rows 2 and 3 each
    hold two separate spans."""
    region = np.zeros((10, 12), dtype=bool)
    region[2:4, 3:5] = True
    region[2:4, 7:9] = True
    region[4:6, 3:9] = True
    return region


def run_climb(metric: Callable[[float], float]) -> tuple[list[ClimbStep], ClimbResult]:
    steps: list[ClimbStep] = []
    result = climb(metric, steps.append, 0.0, SEARCH)
    return steps, result


def test_climb_converges_on_a_peak_inside_the_range() -> None:
    steps, result = run_climb(lambda z: math.exp(-((z - 7.3) ** 2) / 200.0))
    assert abs(result.best_z_um - 7.3) < 1.0
    assert not result.at_edge
    assert steps[0] == ClimbStep(0.0, math.exp(-(7.3**2) / 200.0), 0.0)
    assert all(abs(step.z_um) <= SEARCH.search_range_um for step in steps)


def test_climb_stops_at_the_range_edge_when_still_rising() -> None:
    search = FocusSearchGroup(search_range_um=12.0, initial_step_um=5.0, plateau_pct=1.0)
    visited: list[float] = []
    result = climb(lambda z: z + 100.0, lambda step: visited.append(step.z_um), 0.0, search)
    assert result.at_edge
    assert result.best_z_um == 12.0
    assert visited == [0.0, 5.0, 10.0, 12.0]


def test_climb_on_a_flat_metric_refines_once_then_stops() -> None:
    steps, result = run_climb(lambda z: 1.0)
    assert result.best_z_um == 0.0
    assert not result.at_edge
    assert [step.z_um for step in steps] == [0.0, 5.0, -5.0, 2.5, -2.5]


def test_the_centre_rectangle_is_the_fraction_of_the_grid() -> None:
    scan = ScanGroup(x_pixels=64, y_pixels=64)
    region = roi_region(RoiGroup(mask=None, center_fraction=0.25), scan)
    expected = np.zeros((64, 64), dtype=bool)
    expected[24:40, 24:40] = True
    np.testing.assert_array_equal(region, expected)


def test_a_mask_of_two_separate_areas_is_refused() -> None:
    array = np.zeros((10, 12), dtype=np.uint8)
    array[1:3, 1:3] = 255
    array[6:8, 6:8] = 255
    roi = RoiGroup(mask=MaskRef(source_id="m", array=array), center_fraction=0.25)
    with pytest.raises(ParameterError, match="one connected area, found 2"):
        roi_region(roi, small_scan())


def test_an_empty_mask_is_refused() -> None:
    roi = RoiGroup(mask=MaskRef(source_id="m", array=np.zeros((10, 12), np.uint8)))
    with pytest.raises(ParameterError, match="no pixels"):
        roi_region(roi, small_scan())


def test_rows_sweep_their_hull_plus_the_extra_steps() -> None:
    scan = small_scan()
    spans = row_spans(u_shape())
    assert [(span.y, span.x0, span.x1) for span in spans] == [
        (2, 3, 8),
        (3, 3, 8),
        (4, 3, 8),
        (5, 3, 8),
    ]
    samples_per_pixel = 3
    waveform = roi_waveform(scan, spans, samples_per_pixel)
    swept = 6 + scan.extra_left + scan.extra_right
    assert waveform.shape == (2, len(spans) * swept * samples_per_pixel)

    per_pixel = waveform[:, ::samples_per_pixel]
    for index, span in enumerate(spans):
        row = per_pixel[:, index * swept : (index + 1) * swept]
        for x in range(span.x0, span.x1 + 1):
            column = scan.extra_left + x - span.x0
            assert tuple(row[:, column]) == pytest.approx(scan.voltage_at(x, span.y))


def test_a_region_frame_lands_on_its_bounding_box() -> None:
    """The fast volts, played back as if they were the signal, must land on
    the pixels they were sampled at, with zero wherever no row swept."""
    scan = small_scan()
    region = np.zeros((10, 12), dtype=bool)
    region[3, 5:7] = True
    region[4, 4:9] = True
    spans = row_spans(region)
    box = Box.around(spans)
    waveform = roi_waveform(scan, spans, 2)
    frame = place_rows(waveform[:1], scan, spans, 2, box)

    expected = np.zeros((1, 2, 5), dtype=np.float32)
    for y, x in zip(*np.nonzero(region), strict=True):
        expected[0, y - box.top, x - box.left] = scan.voltage_at(int(x), int(y))[0]
    np.testing.assert_allclose(frame, expected, rtol=1e-6)


def test_the_full_raster_samples_where_voltage_at_says() -> None:
    scan = small_scan()
    waveform = raster_waveform(scan, 1).reshape(2, scan.y_pixels, scan.total_x)
    for x, y in ((0, 0), (5, 3), (11, 9)):
        sampled = waveform[:, y, scan.extra_left + x]
        assert tuple(sampled) == pytest.approx(scan.voltage_at(x, y))


def test_a_sharp_edge_scores_higher_than_a_blurred_one() -> None:
    sharp = np.zeros((1, 32, 32), dtype=np.float32)
    sharp[0, :, 16:] = 1.0
    blurred = cv2.GaussianBlur(sharp[0], (0, 0), 3.0)[None]
    weights = np.ones((32, 32), dtype=bool)
    assert tenengrad(sharp, weights) > tenengrad(blurred, weights)


def test_pixels_outside_the_weights_do_not_count() -> None:
    frame = np.zeros((2, 16, 16), dtype=np.float32)
    frame[:, :, 12:] = 1.0
    weights = np.zeros((16, 16), dtype=bool)
    weights[4:12, 2:8] = True
    assert tenengrad(frame, weights) == 0.0


def test_the_weights_drop_the_region_edge() -> None:
    region = np.zeros((10, 12), dtype=bool)
    region[2:7, 3:9] = True
    box = Box.around(row_spans(region))
    expected = np.zeros((5, 6), dtype=bool)
    expected[1:4, 1:5] = True
    np.testing.assert_array_equal(metric_weights(region, box), expected)


def test_a_region_with_no_interior_is_refused() -> None:
    region = np.zeros((10, 12), dtype=bool)
    region[2:4, 3:9] = True
    with pytest.raises(ParameterError, match="too thin"):
        metric_weights(region, Box.around(row_spans(region)))


def test_a_modulation_mask_keeps_its_line_in_the_session() -> None:
    mask = Mask(source_id="m", source_label="drawn", port=1, line=3)
    assert mask.to_dict() == {"source_id": "m", "source_label": "drawn", "port": 1, "line": 3}
    assert Mask.from_value(mask.to_dict()) == mask
    assert MaskRef(source_id="m", source_label="drawn").to_dict() == {
        "source_id": "m",
        "source_label": "drawn",
    }
