"""Simulated Mosaic: the mosaic, over the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import RunContext, program_registry
from pyrpoc.plugins.devices.simulated import SimulatedScanner

from ...building_blocks.multisite import Microscope, MosaicProgram
from ..dish_microscope import DishMicroscope


@program_registry.register("simulated_mosaic")
class SimulatedMosaic(MosaicProgram):
    display_name = "Mosaic"
    group = "Simulations"
    order = 23
    devices = [SimulatedScanner]

    def microscope(self, ctx: RunContext) -> Microscope:
        return DishMicroscope(ctx)

    def overscan(self, ctx: RunContext) -> tuple[int, int]:
        # The simulated scanner has no mirrors to turn around.
        return (0, 0)
