"""A scan geometry's raster over the Simulated Dish, as the rig's ``Raster`` takes it."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pyrpoc.plugins.devices.simulated import SimulatedDish, SimulatedScanner
from pyrpoc.plugins.program_io.inputs import ScanGeometry


def dish_frame(
    scanner: SimulatedScanner,
    dish: SimulatedDish,
    scan: ScanGeometry,
    sleep: Callable[[float], None],
) -> np.ndarray:
    """One ``(C, H, W)`` frame of ``scan``'s displayed pixels."""
    fast = scan.fast_volts(np.arange(scan.x_pixels))
    slow = scan.slow_volts(np.arange(scan.y_pixels))
    return scanner.scan(dish, fast, slow, scan.dwell_time_us, sleep)
