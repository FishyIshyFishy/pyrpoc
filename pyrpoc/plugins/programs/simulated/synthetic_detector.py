"""The fake detector's pieces: keyed randomness, noise, channel names. Every
program that fakes its data uses them, so fakes stay comparable."""

from __future__ import annotations

import numpy as np

# Keeps the noise generator off the pattern's stream, so noise never moves a feature.
NOISE_STREAM = 0x0125E


def seeded_rng(*parts: int) -> np.random.Generator:
    """A generator keyed by whatever identifies this plane."""
    return np.random.default_rng([part % (2**32) for part in parts])


def with_noise(frame: np.ndarray, noise_level: float, *, seed: int, index: int) -> np.ndarray:
    """``frame`` plus gaussian noise keyed by (seed, index), clipped at zero
    since a detector cannot read negative."""
    if noise_level:
        noise = seeded_rng(seed, index, NOISE_STREAM).standard_normal(frame.shape)
        frame = frame + noise.astype(np.float32) * noise_level
    return np.clip(frame, 0.0, None).astype(np.float32, copy=False)


def sim_labels(channels: int) -> list[str]:
    return [f"sim{index}" for index in range(channels)]
