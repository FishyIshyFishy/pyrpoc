"""How much of a plane is signal: its background level and noise, by robust statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Scales a median absolute deviation to a Gaussian standard deviation.
MAD_TO_SIGMA = 1.4826
# The fraction of pixels taken as a plane's bright end, so one hot pixel cannot carry it.
BRIGHT_QUANTILE = 0.995


@dataclass(frozen=True)
class Floor:
    """A plane's background level and the spread of its noise."""

    level: float
    sigma: float

    def score(self, value: float) -> float:
        """How many noise widths ``value`` stands above the background."""
        return (value - self.level) / self.sigma


def floor_of(plane: np.ndarray) -> Floor:
    """The background and noise of a plane that is mostly background."""
    # The median and its absolute deviation ignore up to half the pixels being signal.
    level = float(np.median(plane))
    spread = float(np.median(np.abs(plane - level))) * MAD_TO_SIGMA
    return Floor(level, max(spread, 1e-9))


def signal_score(plane: np.ndarray) -> float:
    """How far a plane's bright end stands above its noise, in noise widths."""
    return floor_of(plane).score(float(np.quantile(plane, BRIGHT_QUANTILE)))


def strongest_channel(frame: np.ndarray) -> int:
    """The channel of a ``(C, H, W)`` frame with the most signal over its noise."""
    return int(np.argmax([signal_score(plane) for plane in frame]))
