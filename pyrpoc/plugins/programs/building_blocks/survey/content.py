"""Whether a tile holds anything: the channels in which a feature-sized patch stands clear of
the tile's own noise."""

from __future__ import annotations

import cv2
import numpy as np

from ..vision.features import CLEAR_SCORE, MINIMUM_AREA_PX, SMOOTH_SIGMA_PX
from ..vision.noise import floor_of

# How far around content focus is judged, so the blur a feature spreads into still counts.
REACH_PX = 6


def clear_pixels(plane: np.ndarray) -> np.ndarray:
    """The pixels of the smoothed plane that stand ``CLEAR_SCORE`` noise widths above its
    background."""
    # Smoothed first, so the noise is judged at the scale of a feature rather than a pixel.
    smooth = cv2.GaussianBlur(plane.astype(np.float32), (0, 0), SMOOTH_SIGMA_PX)
    floor = floor_of(smooth)
    return smooth > floor.level + CLEAR_SCORE * floor.sigma


def has_content(plane: np.ndarray) -> bool:
    """Whether at least a feature's area of the plane stands clear of its noise."""
    return int(np.count_nonzero(clear_pixels(plane))) >= MINIMUM_AREA_PX


def content_region(frame: np.ndarray, channels: tuple[int, ...]) -> np.ndarray:
    """The pixels near content in any of ``channels``, where focus is judged."""
    # Judged only there, since the noise of an empty part of the tile adds nothing but
    # scatter to the sharpness.
    clear = np.any([clear_pixels(frame[channel]) for channel in channels], axis=0)
    kernel = np.ones((2 * REACH_PX + 1, 2 * REACH_PX + 1), dtype=np.uint8)
    return cv2.dilate(clear.astype(np.uint8), kernel) > 0


def content_channels(frame: np.ndarray) -> tuple[int, ...]:
    """The channels of a ``(C, H, W)`` frame that hold content."""
    return tuple(index for index, plane in enumerate(frame) if has_content(plane))
