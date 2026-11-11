"""Simulated Confocal: the confocal raster, scanned over the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.plugins.devices.simulated import SimulatedDish, SimulatedScanner
from pyrpoc.plugins.program_io.inputs import NUM_FRAMES, SCAN_GEOMETRY_INPUTS, ScanGeometry
from pyrpoc.plugins.program_io.outputs import ImageOutput

from ...building_blocks.functions.run_loops import count_off
from ...building_blocks.runners import Continuous, Single
from ..dish_scan import dish_frame


@program_registry.register("simulated_confocal")
class SimulatedConfocal(Program):
    display_name = "Confocal"
    group = "Simulations"
    order = 22
    devices = [SimulatedScanner]
    inputs = [*SCAN_GEOMETRY_INPUTS, NUM_FRAMES]
    cards = {"Scan": SCAN_GEOMETRY_INPUTS, "Frames": (NUM_FRAMES,)}
    emits = {"intensity": ImageOutput}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        scanner = ctx.devices[SimulatedScanner]
        dish = ctx.devices[SimulatedDish]
        scan = ScanGeometry.from_inputs(ctx.inputs)
        intensity = ctx.output("intensity", ImageOutput)
        for _ in count_off(ctx, ctx.inputs[NUM_FRAMES], "frame"):
            intensity.add(dish_frame(scanner, dish, scan, False, ctx.sleep), scanner.channel_labels)
