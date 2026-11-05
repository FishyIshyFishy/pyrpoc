"""``SpectrumSeriesOutput``: spectra taken one after another, each with when it was taken."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pyrpoc.core.program_output import Origin, Provenance, program_output_registry

from .spectrum import SpectrumOutput


@program_output_registry.register("spectrum_series")
class SpectrumSeriesOutput(SpectrumOutput):
    """A ``SpectrumOutput`` whose spectrum ``i`` was read from ``times_s[i][0]`` to
    ``times_s[i][1]`` seconds after the series began."""

    def __init__(self, *, name: str, provenance: Provenance, origin: Origin, root: Path | None):
        super().__init__(name=name, provenance=provenance, origin=origin, root=root)
        self.times_s: list[tuple[float, float]] = []

    def mark_time(self, start_s: float, end_s: float) -> None:
        """When the spectrum just added was read."""
        with self.lock:
            self.times_s.append((start_s, end_s))

    def extra(self) -> dict[str, Any]:
        return {**super().extra(), "times_s": np.asarray(self.times_s, dtype=np.float64)}

    def take_extra(self, archive: Any) -> None:
        super().take_extra(archive)
        if "times_s" in archive:
            self.times_s = [(float(start), float(end)) for start, end in archive["times_s"]]
