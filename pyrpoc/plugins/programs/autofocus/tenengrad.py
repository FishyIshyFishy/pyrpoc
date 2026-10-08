"""The focus metric: Tenengrad, the summed squared Sobel gradient."""

from __future__ import annotations

import cv2
import numpy as np


def tenengrad(frame: np.ndarray, weights: np.ndarray) -> float:
    """Each channel's Tenengrad over the ``weights`` pixels of a ``(C, H, W)``
    frame, summed across channels. Per channel, because the gradient is squared:
    the Tenengrad of a channel sum would add cross-terms between detectors."""
    total = 0.0
    for plane in np.ascontiguousarray(frame, dtype=np.float32):
        gx = cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3)
        total += float(np.sum((gx * gx + gy * gy)[weights], dtype=np.float64))
    return total
