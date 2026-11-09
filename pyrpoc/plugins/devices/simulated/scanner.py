"""The Simulated Scanner: a galvo raster over the Simulated Dish it is ``backed_by``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pyrpoc.core.device import Device, DeviceConfig
from pyrpoc.utils.simulation import CHANNELS

from .dish import SimulatedDish


@dataclass
class ScannerConfig(DeviceConfig):
    pass


class SimulatedScanner(Device):
    display_name = "Simulated Scanner"
    backed_by = SimulatedDish
    config_cls = ScannerConfig

    config: ScannerConfig

    @property
    def channel_labels(self) -> list[str]:
        return [f"sim{index}" for index in range(CHANNELS)]

    def scan(
        self,
        dish: SimulatedDish,
        fast_volts: np.ndarray,
        slow_volts: np.ndarray,
        dwell_us: float,
        treating: bool,
        sleep: Callable[[float], None],
    ) -> np.ndarray:
        """A ``(C, H, W)`` frame of the galvo held at each (fast, slow) volt pair for
        ``dwell_us``, with the treatment light gated on or off, taking as long as the scan
        would."""
        frame = dish.dish.scan(fast_volts, slow_volts, dwell_us, treating)
        dish.wait(len(fast_volts) * len(slow_volts) * dwell_us * 1e-6, sleep)
        return frame
