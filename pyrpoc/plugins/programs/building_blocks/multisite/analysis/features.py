"""Finding one clear feature in a plane: a bright connected region that stands well above the
noise."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .noise import floor_of

# Five noise widths is the usual bar for calling something detected rather than noise.
CLEAR_SCORE = 5.0
MINIMUM_AREA_PX = 25
SMOOTH_SIGMA_PX = 1.5


@dataclass(frozen=True)
class Box:
    """A region of a plane, in pixels, ``bottom`` and ``right`` exclusive."""

    top: int
    left: int
    bottom: int
    right: int

    @property
    def centre(self) -> tuple[float, float]:
        return (self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0

    def grown(self, margin: int, shape: tuple[int, int]) -> Box:
        height, width = shape
        return Box(
            max(0, self.top - margin),
            max(0, self.left - margin),
            min(height, self.bottom + margin),
            min(width, self.right + margin),
        )

    def crop(self, plane: np.ndarray) -> np.ndarray:
        return plane[self.top : self.bottom, self.left : self.right]


def bright_regions(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Connected regions above Otsu's threshold of the smoothed plane: their label image
    and their ``cv2`` stats."""
    smooth = cv2.GaussianBlur(plane.astype(np.float32), (0, 0), SMOOTH_SIGMA_PX)
    span = float(smooth.max() - smooth.min()) or 1.0
    scaled = ((smooth - smooth.min()) / span * 255.0).astype(np.uint8)
    _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return labels, stats


def clear_features(plane: np.ndarray, within: Box) -> list[Box]:
    """The bright regions whose centre lies in ``within``, which do not touch the plane's edge
    and stand ``CLEAR_SCORE`` noise widths above the background, clearest first."""
    floor = floor_of(plane)
    labels, stats = bright_regions(plane)
    height, width = plane.shape
    found: list[tuple[float, Box]] = []
    for index in range(1, len(stats)):
        left, top, w, h, area = (int(value) for value in stats[index])
        box = Box(top, left, top + h, left + w)
        centre_x, centre_y = box.centre
        inside = within.left <= centre_x < within.right and within.top <= centre_y < within.bottom
        touches = top == 0 or left == 0 or box.bottom == height or box.right == width
        if area < MINIMUM_AREA_PX or touches or not inside:
            continue
        score = floor.score(float(plane[labels == index].mean()))
        if score >= CLEAR_SCORE:
            found.append((score, box))
    return [box for _score, box in sorted(found, key=lambda item: -item[0])]


def clear_feature(plane: np.ndarray, within: Box) -> Box | None:
    """The clearest of ``clear_features``, or None if there is none."""
    features = clear_features(plane, within)
    return features[0] if features else None
