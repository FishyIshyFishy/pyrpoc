"""The segmentation: an Otsu threshold, opened to drop single-pixel speckle."""

from __future__ import annotations

import cv2
import numpy as np

# Small enough to keep thin structures, big enough to drop shot-noise pixels.
OPEN_KERNEL = np.ones((3, 3), dtype=np.uint8)


def otsu_mask(plane: np.ndarray) -> np.ndarray:
    """A ``(H, W)`` uint8 mask of ``plane``, 255 above its Otsu threshold.
    Scaled to 8 bits first, since cv2's Otsu takes only integer images."""
    lo = float(np.min(plane))
    hi = float(np.max(plane))
    if hi <= lo:
        return np.zeros(plane.shape, dtype=np.uint8)
    scaled = ((plane - lo) * (255.0 / (hi - lo))).astype(np.uint8)
    _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_KERNEL)
