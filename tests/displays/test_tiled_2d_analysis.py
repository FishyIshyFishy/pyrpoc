"""Tests for the tiled 2D display's ImageJ-style analysis suite.

The statistics layer is pure numpy and tested directly. The widget-level tests
drive the display headlessly (offscreen Qt) to exercise tool selection, the
cross-channel computations, and persistence round-tripping of the analysis state.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.displays.tiled_2d.display import Tiled2DDisplay
from pyrpoc.displays.tiled_2d.statistics import (
    ellipse_mask,
    line_profile,
    pixel_value,
    rectangle_mask,
    region_stats,
)
from pyrpoc.displays.tiled_2d.toolbar import AnalysisTool


# ----------------------------------------------------------------------
# Pure statistics
# ----------------------------------------------------------------------


def test_rectangle_mask_selects_expected_pixels():
    mask = rectangle_mask(10, 10, x=2, y=3, w=4, h=2)
    assert mask.sum() == 8
    assert mask[3, 2] and mask[4, 5]
    assert not mask[2, 2] and not mask[5, 2]


def test_rectangle_mask_clips_to_bounds():
    mask = rectangle_mask(5, 5, x=-3, y=-3, w=4, h=4)
    assert mask.sum() == 1
    assert mask[0, 0]


def test_ellipse_mask_is_inscribed_and_smaller_than_box():
    full = rectangle_mask(40, 40, x=5, y=5, w=30, h=30)
    oval = ellipse_mask(40, 40, x=5, y=5, w=30, h=30)
    assert oval.sum() < full.sum()
    # Center pixel inside, box corner outside the inscribed oval.
    assert oval[20, 20]
    assert not oval[5, 5]


def test_region_stats_matches_numpy():
    channel = np.arange(25, dtype=np.float32).reshape(5, 5)
    mask = rectangle_mask(5, 5, x=0, y=0, w=2, h=2)
    stats = region_stats(channel, mask)
    expected = channel[:2, :2]
    assert stats.count == 4
    assert stats.mean == pytest.approx(float(expected.mean()))
    assert stats.maximum == pytest.approx(float(expected.max()))
    assert stats.total == pytest.approx(float(expected.sum()))


def test_region_stats_empty_mask():
    channel = np.ones((4, 4), dtype=np.float32)
    stats = region_stats(channel, np.zeros((4, 4), dtype=bool))
    assert stats.count == 0
    assert stats.mean == 0.0


def test_line_profile_reads_horizontal_gradient():
    channel = np.tile(np.arange(10, dtype=np.float32), (10, 1))
    distances, values = line_profile(channel, 0.0, 5.0, 9.0, 5.0)
    assert distances[0] == 0.0
    assert distances[-1] == pytest.approx(9.0)
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(9.0)


def test_pixel_value_inside_and_outside():
    channel = np.arange(9, dtype=np.float32).reshape(3, 3)
    assert pixel_value(channel, 1.4, 2.1) == pytest.approx(7.0)
    assert pixel_value(channel, -1.0, 0.0) is None
    assert pixel_value(channel, 3.0, 0.0) is None


# ----------------------------------------------------------------------
# Widget-level behaviour
# ----------------------------------------------------------------------


@pytest.fixture
def display(qapp, chw_frame):
    widget = Tiled2DDisplay()
    widget.set_data(chw_frame)
    yield widget
    widget.deleteLater()


def test_rectangle_tool_creates_roi_and_fills_stats_table(display, chw_frame):
    display.on_tool_changed(AnalysisTool.RECTANGLE)
    assert display.controller._roi is not None
    assert not display.panel.isHidden()
    # One stats row per channel.
    assert display.panel._table.rowCount() == chw_frame.shape[0]


def test_line_tool_plots_one_curve_per_channel(display, chw_frame):
    display.on_tool_changed(AnalysisTool.LINE)
    assert display.controller._roi is not None
    assert len(display.panel._profile_curves) == chw_frame.shape[0]


def test_inspect_tool_reports_pixel_values(display):
    display.on_tool_changed(AnalysisTool.INSPECT)
    display.controller.update_inspector(0, 3.0, 4.0)
    text = display.panel._readout.text()
    assert "x = 3" in text
    assert "y = 4" in text


def test_clear_removes_overlay_and_returns_to_pan(display):
    display.on_tool_changed(AnalysisTool.RECTANGLE)
    display.on_clear_requested()
    assert display.controller._roi is None
    assert display.controller.current_tool() == AnalysisTool.PAN
    assert display.panel.isHidden()


def test_switching_tools_replaces_the_roi(display):
    display.on_tool_changed(AnalysisTool.RECTANGLE)
    first = display.controller._roi
    display.on_tool_changed(AnalysisTool.ELLIPSE)
    second = display.controller._roi
    assert second is not None and second is not first


def test_analysis_state_survives_persistence_round_trip(qapp, chw_frame):
    source = Tiled2DDisplay()
    source.set_data(chw_frame)
    source.on_tool_changed(AnalysisTool.RECTANGLE)
    geometry = source.controller.roi_geometry()
    state = source.export_persistence_state()
    source.deleteLater()

    assert state["analysis"]["tool"] == "rectangle"

    restored = Tiled2DDisplay()
    restored.import_persistence_state(state)
    # ROI is recreated once tiles/data exist (first frame).
    restored.set_data(chw_frame)
    assert restored.controller.current_tool() == AnalysisTool.RECTANGLE
    assert restored.controller._roi is not None
    restored_geometry = restored.controller.roi_geometry()
    assert restored_geometry["pos"] == pytest.approx(geometry["pos"])
    assert restored_geometry["size"] == pytest.approx(geometry["size"])
    restored.deleteLater()


def test_line_geometry_round_trips(qapp, chw_frame):
    source = Tiled2DDisplay()
    source.set_data(chw_frame)
    source.on_tool_changed(AnalysisTool.LINE)
    geometry = source.controller.roi_geometry()
    state = source.export_persistence_state()
    source.deleteLater()

    restored = Tiled2DDisplay()
    restored.import_persistence_state(state)
    restored.set_data(chw_frame)
    assert restored.controller.current_tool() == AnalysisTool.LINE
    np.testing.assert_allclose(
        np.array(restored.controller.roi_geometry()["points"]),
        np.array(geometry["points"]),
        atol=1e-4,
    )
    restored.deleteLater()
