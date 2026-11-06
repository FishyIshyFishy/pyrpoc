"""``CalibrationSink``: where a calibration shows each frame it takes and says what it did."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from pyrpoc.plugins.program_io.outputs import CalibrationEvent


class CalibrationSink(Protocol):
    """Shown every frame as it is taken, and told each event as it happens."""

    def show(self, frame: np.ndarray) -> None: ...

    def log(self, event: CalibrationEvent) -> None: ...
