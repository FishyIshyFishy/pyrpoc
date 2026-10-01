"""A pixel's samples split into a t0 and a t2 window, with t1 discarded between."""

from __future__ import annotations

import numpy as np

from ..building_blocks.parameter_groups import SplitGroup


def gated_to_t0(
    ttl: dict[str, np.ndarray], samples_per_pixel: int, t0_samples: int
) -> dict[str, np.ndarray]:
    """Each mask's TTL high only in the first ``t0_samples`` of every pixel."""
    gated: dict[str, np.ndarray] = {}
    for channel, signal in ttl.items():
        per_pixel = signal.reshape(-1, samples_per_pixel).copy()
        per_pixel[:, t0_samples:] = False
        gated[channel] = per_pixel.reshape(-1)
    return gated


def split_frame(samples: np.ndarray, split: SplitGroup) -> np.ndarray:
    """``(C, H, W, S)`` samples as a ``(C*2, H, W)`` frame, the t0 and t2 means
    of each channel interleaved. A t2 window with no samples left reads zero."""
    early = samples[..., : split.t0_samples].mean(axis=3)
    late_start = split.t0_samples + split.t1_samples
    if late_start < samples.shape[3]:
        late = samples[..., late_start:].mean(axis=3)
    else:
        late = np.zeros_like(early)
    channels, height, width = early.shape
    return np.stack((early, late), axis=1).reshape(channels * 2, height, width)


def split_labels(labels: list[str]) -> list[str]:
    """``ai0_t0``, ``ai0_t2``, ``ai1_t0``, ...: interleaved, matching the frame."""
    return [f"{label}_{window}" for label in labels for window in ("t0", "t2")]
