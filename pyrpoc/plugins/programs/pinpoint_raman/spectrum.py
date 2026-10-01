"""The synthetic spectrum published until there is a CCD device."""

from __future__ import annotations

import numpy as np

from ..building_blocks.functions.synthetic_detector import seeded_rng
from ..building_blocks.parameter_groups import Point, SpectrumGroup

# Keeps the noise generator off the band generator's stream, so changing the
# frame index cannot shift a band centre.
NOISE_STREAM = 0x5EED


def synthetic_spectrum(spec: SpectrumGroup, point: Point, *, frame_index: int) -> np.ndarray:
    """A ``(1, n_points)`` spectrum of gaussian bands, keyed to the position.

    The position enters the seed quantised to 0.1 mV, so a re-click reproduces
    the trace and a click one pixel over does not. Only the noise varies per
    frame, so a long run looks like a detector integrating.
    """
    bands = seeded_rng(spec.seed, round(point.fast_v * 1e4), round(point.slow_v * 1e4))
    xs = np.arange(spec.n_points, dtype=np.float32)
    spectrum = np.zeros(spec.n_points, dtype=np.float32)
    centres = bands.uniform(0.05, 0.95, spec.n_peaks) * spec.n_points
    widths = bands.uniform(0.004, 0.02, spec.n_peaks) * spec.n_points
    heights = bands.uniform(0.2, 1.0, spec.n_peaks)
    for centre, width, height in zip(centres, widths, heights, strict=True):
        spectrum += height * np.exp(-((xs - centre) ** 2) / (2.0 * width**2))

    # A broad fluorescence background, so the bands sit on something.
    spectrum += 0.12 * np.exp(-xs / (0.6 * spec.n_points))

    noise = seeded_rng(spec.seed, frame_index, NOISE_STREAM)
    spectrum = spectrum + noise.normal(0.0, spec.noise_level, spec.n_points)
    return np.clip(spectrum, 0.0, None).astype(np.float32)[np.newaxis]
