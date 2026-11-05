"""Simulated Confocal: the confocal raster, scanned over the Simulated Dish."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices.simulated import SimulatedDish, SimulatedScanner
from pyrpoc.plugins.program_io.inputs import NUM_FRAMES, SCAN_GEOMETRY_INPUTS, ScanGeometry
from pyrpoc.plugins.program_io.outputs import Image

from ...building_blocks.functions.run_loops import count_off
from ...building_blocks.runners import Continuous, Single
from ..dish_scan import dish_frame


class SimulatedConfocal(Program):
    display_name = "Confocal"
    group = "Simulations"
    order = 21
    devices = [SimulatedScanner]
    sections = [
        InputSection("Scan", SCAN_GEOMETRY_INPUTS),
        InputSection("Frames", (NUM_FRAMES,)),
    ]
    emits = {"intensity": Image}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        scanner = ctx.devices[SimulatedScanner]
        dish = ctx.devices[SimulatedDish]
        scan = ScanGeometry.from_inputs(ctx.inputs)
        for _ in count_off(ctx, ctx.inputs[NUM_FRAMES], "frame"):
            frame = dish_frame(scanner, dish, scan, ctx.sleep)
            ctx.publish("intensity", Image(frame, scanner.channel_labels))
