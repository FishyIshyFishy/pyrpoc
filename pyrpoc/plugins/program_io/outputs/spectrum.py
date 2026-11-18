"""``SpectrumOutput``: intensity per wavelength bin, one spectrum per channel, and the exposure
the detector actually used."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyrpoc.core.program_output import program_output_registry
from pyrpoc.core.recording import Recording

from .building_blocks.npz import NpzStack


@program_output_registry.register("spectrum")
class SpectrumOutput(NpzStack):
    """``(channel, wavelength)`` spectra. Bins carry no units: that is the spectrometer's
    calibration."""

    axes = ("channel", "wavelength")

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        # Seconds per frame as the detector rounded it, which may differ from what was asked.
        self.exposure_s: float | None = None

    def set_exposure(self, seconds: float) -> None:
        with self.lock:
            self.exposure_s = seconds

    def extra(self) -> dict[str, np.ndarray]:
        return {} if self.exposure_s is None else {"exposure_s": np.asarray(self.exposure_s)}

    def take_extra(self, arrays: dict[str, np.ndarray]) -> None:
        if "exposure_s" in arrays:
            self.exposure_s = float(arrays["exposure_s"])
