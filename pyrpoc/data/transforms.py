"""Array transforms shared by whoever renders or re-uses acquired data.

``normalize_channels`` was duplicated as ``get_normalized_data_3d`` in both
displays. Only views/ uses it today, so by section 6.4's test this is the file
in data/ most likely to be misfiled; section 5 places it here and programs are
the intended second caller. Worth revisiting in 3.2 if nothing outside views/
has picked it up.
"""

from __future__ import annotations

import numpy as np


def normalize_channels(data_chw: np.ndarray | None) -> np.ndarray | None:
    """Per-channel min/max scaling of a ``(C, H, W)`` array into [0, 1].

    A flat channel (max <= min) stays all zeros, as it did in v3.0.
    """
    if data_chw is None:
        return None
    arr = np.asarray(data_chw, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected channels-first [C, H, W], got shape {arr.shape}")

    norm = np.zeros_like(arr, dtype=np.float32)
    for index in range(arr.shape[0]):
        channel = arr[index]
        lo = float(np.min(channel))
        hi = float(np.max(channel))
        if hi > lo:
            norm[index] = (channel - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


def channel_levels(channel: np.ndarray) -> tuple[float, float]:
    """Autoscale bounds for one channel, never degenerate."""
    lo = float(np.min(channel))
    hi = float(np.max(channel))
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi
