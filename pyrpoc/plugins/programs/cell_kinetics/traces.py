"""Intensity traces over one channel of a series: a region's mean, less the
background's, normalised to the first frame (F/F0)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Keeps F/F0 finite when a region fades to the background.
FLOOR = 1e-4


def otsu(frame: np.ndarray) -> float:
    """The intensity that best splits ``frame`` into two classes (Otsu's
    method): fluorescent pixels above it, background below."""
    counts, edges = np.histogram(frame, bins=256)
    centres = (edges[:-1] + edges[1:]) / 2
    below = np.cumsum(counts)
    above = below[-1] - below
    summed = np.cumsum(counts * centres)
    mean_below = summed / np.maximum(below, 1)
    mean_above = (summed[-1] - summed) / np.maximum(above, 1)
    spread = below * above * (mean_below - mean_above) ** 2
    return float(edges[1:][np.argmax(spread)])


@dataclass
class Regions:
    """One channel's pixels, split by its own first frame and the mask."""

    treated: np.ndarray
    control: np.ndarray
    background: np.ndarray
    threshold: float

    @classmethod
    def split(cls, frame: np.ndarray, mask: np.ndarray) -> Regions:
        threshold = otsu(frame)
        fluorescent = frame > threshold
        return cls(fluorescent & mask, fluorescent & ~mask, ~fluorescent, threshold)

    def describe(self) -> dict[str, float]:
        """JSON-safe, for the output's metadata."""
        return {
            "treated_px": int(self.treated.sum()),
            "control_px": int(self.control.sum()),
            "background_px": int(self.background.sum()),
            "threshold": self.threshold,
        }


def normalized(stack: np.ndarray, region: np.ndarray, background: np.ndarray) -> np.ndarray:
    """``region``'s background-subtracted trace as F/F0. ``stack`` is ``(T, H, W)``."""
    net = np.maximum(stack[:, region].mean(axis=1) - stack[:, background].mean(axis=1), FLOOR)
    return net / net[0]


def corrected(treated: np.ndarray, control: np.ndarray) -> np.ndarray:
    """``treated`` with the control's decay divided out, leaving what the
    treatment did beyond what imaging alone does."""
    return treated / np.clip(control, FLOOR, None)
