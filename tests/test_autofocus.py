"""Autofocus without a rig: the z search, the region scan's geometry, and the
focus metric, each checked against a case whose answer is known."""

from __future__ import annotations

import math
from collections.abc import Callable

import cv2
import numpy as np
import pytest

from pyrpoc.plugins.programs.autofocus.roi_scan import (
    Box,
    metric_weights,
    place_rows,
    roi_region,
    roi_waveform,
    row_spans,
)
from pyrpoc.plugins.programs.autofocus.search import (
    FocusResult,
    FocusStep,
    fit_peak,
    search_focus,
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

SEARCH = FocusSearchGroup(
    search_range_um=50.0, coarse_step_um=5.0, fine_step_um=0.5, min_contrast=2.0
)


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


class Drive:
    """A focus drive over a known curve, recording every z it is sent to."""

    def __init__(self, curve: Callable[[float], float]):
        self.curve = curve
        self.moves: list[float] = []
        self.steps: list[FocusStep] = []

    def measure(self, z_um: float) -> float:
        self.moves.append(z_um)
        return self.curve(z_um)

    def search(self, search: FocusSearchGroup) -> FocusResult | None:
        return search_focus(self.measure, self.moves.append, self.steps.append, 0.0, search)


def focus_at(
    center_um: float, width_um: float, rng: np.random.Generator
) -> Callable[[float], float]:
    """A Gaussian focus peak over a floor, with a little noise on each frame."""

    def curve(z_um: float) -> float:
        peak = 10.0 * math.exp(-((z_um - center_um) ** 2) / (2.0 * width_um**2))
        return 1.0 + peak + 0.02 * float(rng.standard_normal())

    return curve


def test_a_focus_far_from_the_start_is_found() -> None:
    drive = Drive(focus_at(30.0, 3.0, np.random.default_rng(0)))
    result = drive.search(SEARCH)
    assert result is not None
    assert abs(result.best_z_um - 30.0) < SEARCH.fine_step_um / 2.0
    assert not result.at_edge
    assert all(abs(z) <= SEARCH.search_range_um for z in drive.moves)


def test_a_focus_near_the_start_stops_the_outward_search_early() -> None:
    drive = Drive(focus_at(2.0, 3.0, np.random.default_rng(1)))
    result = drive.search(SEARCH)
    assert result is not None
    bracket = [step for step in drive.steps if step.phase == "bracket"]
    # A full coarse sweep of the range would be 21 frames.
    assert len(bracket) <= 5


def test_no_focus_in_the_range_returns_to_the_start() -> None:
    rng = np.random.default_rng(2)
    drive = Drive(lambda z: 1.0 + 0.02 * float(rng.standard_normal()))
    assert drive.search(SEARCH) is None
    assert drive.moves[-1] == 0.0
    assert {step.phase for step in drive.steps} == {"bracket"}


def test_a_peak_beyond_the_range_stops_at_its_edge() -> None:
    search = FocusSearchGroup(
        search_range_um=12.0, coarse_step_um=5.0, fine_step_um=0.5, min_contrast=2.0
    )
    result = Drive(lambda z: math.exp(z / 3.0)).search(search)
    assert result is not None
    assert result.at_edge
    assert result.best_z_um == 12.0


def test_of_two_peaks_the_nearer_is_chosen() -> None:
    near = focus_at(-12.0, 3.0, np.random.default_rng(3))
    far = focus_at(35.0, 3.0, np.random.default_rng(4))
    result = Drive(lambda z: near(z) + far(z) - 1.0).search(SEARCH)
    assert result is not None
    assert abs(result.best_z_um + 12.0) < SEARCH.fine_step_um


def test_the_fine_sweep_and_the_final_move_only_go_up() -> None:
    drive = Drive(focus_at(-8.0, 3.0, np.random.default_rng(5)))
    result = drive.search(SEARCH)
    assert result is not None
    refine = [step.z_um for step in drive.steps if step.phase == "refine"]
    assert refine == sorted(refine)
    assert drive.moves[-2] < drive.moves[-1] == result.best_z_um


def test_fine_must_be_finer_than_coarse() -> None:
    search = FocusSearchGroup(
        search_range_um=50.0, coarse_step_um=1.0, fine_step_um=1.0, min_contrast=2.0
    )
    with pytest.raises(ParameterError, match="fine step"):
        Drive(lambda z: 1.0).search(search)


def test_the_fit_recovers_a_gaussian_vertex() -> None:
    steps = [
        FocusStep(z, 5.0 * math.exp(-((z - 1.23) ** 2) / 8.0), "refine")
        for z in np.arange(-4.0, 6.0, 0.5)
    ]
    vertex = fit_peak(steps)
    assert vertex is not None
    assert vertex == pytest.approx(1.23, abs=1e-6)


def test_the_fit_refuses_a_flat_or_falling_curve() -> None:
    assert fit_peak([FocusStep(z, 1.0, "refine") for z in (0.0, 1.0, 2.0, 3.0)]) is None
    assert fit_peak([FocusStep(z, 10.0 - z, "refine") for z in (0.0, 1.0, 2.0, 3.0)]) is None


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
