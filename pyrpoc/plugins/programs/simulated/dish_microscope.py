"""``DishMicroscope``: the Simulated Scanner and Dish, as a ``Microscope``."""

from __future__ import annotations

import numpy as np

from pyrpoc.core.program import RunContext
from pyrpoc.plugins.devices.simulated import SimulatedDish, SimulatedScanner
from pyrpoc.plugins.program_io.inputs import ScanGeometry

from ..building_blocks.multisite import DigitalLine
from .dish_scan import dish_frame


class DishMicroscope:
    """The simulated instrument a run claimed, living by the dish's clock."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.dish = ctx.devices[SimulatedDish]
        self.scanner = ctx.devices[SimulatedScanner]

    @property
    def channel_labels(self) -> list[str]:
        return self.scanner.channel_labels

    def scan(self, scan: ScanGeometry, lines: tuple[DigitalLine, ...]) -> np.ndarray:
        self.ctx.check_cancel()
        # A driven line gates the dish's treatment light, wherever it is wired.
        return dish_frame(self.scanner, self.dish, scan, bool(lines), self.ctx.sleep)

    def xy(self) -> tuple[float, float]:
        return self.dish.xy_position()

    def z(self) -> float:
        return self.dish.z_position()

    def move_xy(self, x_um: float, y_um: float) -> None:
        self.dish.move_xy_to(x_um, y_um)
        self.dish.wait_until_stopped("xy", self.ctx.sleep)

    def move_z(self, z_um: float) -> None:
        self.dish.move_z_to(z_um)
        self.dish.wait_until_stopped("z", self.ctx.sleep)

    def now(self) -> float:
        return self.dish.now()

    def wait(self, seconds: float) -> None:
        self.dish.wait(seconds, self.ctx.sleep)

    def stop(self) -> None:
        self.dish.stop_smoothly()
