"""Simulated instruments: produce toy data so the app runs with no hardware.

Entirely separate from the real instruments — the real path never touches this.
"""

from __future__ import annotations

import numpy as np

from pyrpoc_next.instruments.base import Instrument
from pyrpoc_next.instruments.registry import instrument_registry
from pyrpoc_next.instruments.scanning import RasterScan
from pyrpoc_next.instruments.toy import toy_flim_cube, toy_frame
from pyrpoc_next.structs.keys import InstrumentKey


@instrument_registry.register
class SimulatedDAQ(Instrument):
    """A stand-in DAQ that returns toy raster samples instead of reading hardware."""

    key = InstrumentKey.simulated_daq
    display_name = "Simulated DAQ"

    def __init__(self):
        super().__init__()
        self.frame_index = 0

    def run(self, scan: RasterScan, ttl_signals: dict | None = None) -> np.ndarray:
        """Return a toy per-pixel sample cube (channels, H, W, samples)."""
        frame = toy_frame(scan.x_pixels, scan.y_pixels, len(scan.ai_channels), self.frame_index)
        self.frame_index += 1
        return np.repeat(frame[:, :, :, None], scan.pixel_samples, axis=3).astype(np.float32, copy=False)


@instrument_registry.register
class SimulatedTagger(Instrument):
    """A stand-in TimeTagger that returns toy decay-histogram cubes."""

    key = InstrumentKey.simulated_tagger
    display_name = "Simulated TimeTagger"

    def __init__(self):
        super().__init__()
        self.frame_index = 0

    def flim_frame(self, x_pixels: int, y_pixels: int, n_bins: int, bin_width_ps: float,
                   laser_period_ps: float) -> np.ndarray:
        """Return a toy (H, W, bins) histogram cube."""
        cube = toy_flim_cube(x_pixels, y_pixels, n_bins, bin_width_ps, laser_period_ps, self.frame_index)
        self.frame_index += 1
        return cube
