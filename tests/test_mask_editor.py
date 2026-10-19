"""The mask editor's masks: a shape drawn without thresholds is filled whole,
so it reaches a region scan as the one area it was drawn as."""

from __future__ import annotations

import cv2
import numpy as np

from pyrpoc.plugins.data_panels.mask_editor.canvas import MaskRoi, Threshold
from pyrpoc.plugins.data_panels.mask_editor.panel import mask_from_rois


def noisy_frame() -> np.ndarray:
    """A one-channel ``(1, H, W)`` frame of noise across 0..255, like a dim
    fluorescence image after the editor's scaling."""
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 255.0, (1, 64, 64)).astype(np.float32)


def ellipse_points() -> list[tuple[float, float]]:
    angles = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
    return [(32.0 + 20.0 * np.cos(a), 32.0 + 14.0 * np.sin(a)) for a in angles]


def areas(mask: np.ndarray) -> int:
    # The count includes the background's label.
    labels, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return labels - 1


def test_a_shape_without_thresholds_is_filled_whole() -> None:
    points = ellipse_points()
    mask = mask_from_rois([MaskRoi(points=points, threshold=None)], noisy_frame())

    polygon = np.array([[round(x), round(y)] for x, y in points], dtype=np.int32)
    expected = np.zeros((64, 64), dtype=np.uint8)
    cv2.fillPoly(expected, [polygon.reshape(-1, 1, 2)], 255)
    np.testing.assert_array_equal(mask, expected)
    assert areas(mask) == 1


def test_a_threshold_keeps_only_in_range_pixels_of_the_shape() -> None:
    frame = noisy_frame()
    threshold = Threshold(low=51.0, high=204.0, active_channels=[True])
    mask = mask_from_rois([MaskRoi(points=ellipse_points(), threshold=threshold)], frame)

    lit = mask > 0
    assert np.all((frame[0][lit] >= 51.0) & (frame[0][lit] <= 204.0))
    # Over noise the kept pixels scatter: why thresholds are opt-in.
    assert areas(mask) > 1


def test_an_inactive_channel_keeps_nothing() -> None:
    threshold = Threshold(low=0.0, high=255.0, active_channels=[False])
    mask = mask_from_rois([MaskRoi(points=ellipse_points(), threshold=threshold)], noisy_frame())
    assert not mask.any()
