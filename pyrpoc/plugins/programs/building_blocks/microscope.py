"""``Microscope``: what the autonomous programs ask of an instrument, so the same logic runs on
the rig and on the simulated dish."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pyrpoc.plugins.program_io.inputs import ScanGeometry


@dataclass(frozen=True)
class DigitalLine:
    """A DAQ port and line, driven high over a whole scan."""

    port: int
    line: int


class Microscope(Protocol):
    """Scan, move the stage, and keep time. Moves return once the stage has stopped, and every
    wait returns early if the run is stopped."""

    @property
    def channel_labels(self) -> list[str]: ...

    def scan(self, scan: ScanGeometry, lines: tuple[DigitalLine, ...]) -> np.ndarray:
        """One ``(C, H, W)`` frame of ``scan``'s displayed pixels, with ``lines`` driven high
        for the whole scan."""
        ...

    def xy(self) -> tuple[float, float]: ...

    def z(self) -> float: ...

    def move_xy(self, x_um: float, y_um: float) -> None: ...

    def move_z(self, z_um: float) -> None: ...

    def now(self) -> float:
        """Seconds on the clock the sample lives by."""
        ...

    def wait(self, seconds: float) -> None: ...

    def stop(self) -> None:
        """Stop the stage where it is."""
        ...
