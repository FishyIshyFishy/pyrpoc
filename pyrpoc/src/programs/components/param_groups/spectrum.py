"""The stand-in spectrometer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.src.structs.params import Group, float_field, int_field
from pyrpoc.src.structs.registries import block


@block
@dataclass
class SpectrumGroup(Group):
    """A stand-in spectrometer, until there is a real one to configure.

    Deterministic in the same way ``SignalGroup`` is: the spectrum is a function
    of (seed, point, frame index), so the same spot gives the same trace on
    every run and two different spots visibly differ. ``num_frames`` is here
    rather than in a block of its own for the same reason it is in
    ``ScanGroup`` -- it decides how the acquisition is done.
    """

    label: ClassVar[str] = "Spectrum"

    num_frames: int = int_field("Frames", 1, minimum=1, tooltip="Number of spectra to capture")
    integration_ms: int = int_field(
        "Integration (ms)",
        500,
        minimum=0,
        tooltip="Dwell per spectrum, standing in for detector integration time",
    )
    n_points: int = int_field(
        "Points", 1024, minimum=16, maximum=65536, tooltip="Samples along the spectral axis"
    )
    n_peaks: int = int_field(
        "Peaks", 5, minimum=0, maximum=64, tooltip="How many bands to synthesise"
    )
    noise_level: float = float_field(
        "Noise Level", 0.03, minimum=0.0, step=0.01, tooltip="Gaussian noise added per point"
    )
    seed: int = int_field(
        "Seed", 1234, minimum=0, tooltip="Same seed and point give the same spectrum"
    )
