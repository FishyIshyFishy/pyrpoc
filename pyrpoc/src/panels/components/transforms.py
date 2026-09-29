"""Array transforms for panels that render or re-use acquired data."""

from __future__ import annotations

import numpy as np


def normalize_channels(data_chw: np.ndarray) -> np.ndarray:
    """Per-channel min/max scaling of a ``(C, H, W)`` array into [0, 1]. A
    flat channel stays all zeros."""
    arr = np.asarray(data_chw, dtype=np.float32)
    norm = np.zeros_like(arr, dtype=np.float32)
    for index in range(arr.shape[0]):
        channel = arr[index]
        lo = float(np.min(channel))
        hi = float(np.max(channel))
        if hi > lo:
            norm[index] = (channel - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)
