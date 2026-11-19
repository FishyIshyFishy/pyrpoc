"""Confocal (New RPOC): the confocal raster with port 0 driven as the LabVIEW imager drives
it, mask and status lines together, and its static lines set before the scan."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.plugins.devices import DAQ, Galvo
from pyrpoc.plugins.program_io.inputs import (
    MASK_LINES,
    MASKS,
    NUM_FRAMES,
    SAMPLE_RATE_MHZ,
    STATIC_LINES,
    STATUS_LINES,
)
from pyrpoc.plugins.program_io.outputs import ImageOutput
from pyrpoc.plugins.programs.building_blocks.settings import SCAN_GEOMETRY_INPUTS

from ..building_blocks.functions.daq_tasks import hold_static_lines
from ..building_blocks.functions.port_raster import PortRaster
from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.runners import Continuous, Single


@program_registry.register("confocal_new_rpoc")
class ConfocalNewRpoc(Program):
    display_name = "Confocal (New RPOC)"
    group = "Confocal Fluorescence"
    order = 12
    devices = [Galvo, DAQ]
    inputs = [
        *SCAN_GEOMETRY_INPUTS,
        NUM_FRAMES,
        SAMPLE_RATE_MHZ,
        MASKS,
        MASK_LINES,
        *STATUS_LINES,
        *STATIC_LINES,
    ]
    # MASKS has no row: masks are drawn and edited in a Mask Editor.
    cards = {
        "Scan": SCAN_GEOMETRY_INPUTS,
        "Frames": (NUM_FRAMES,),
        "DAQ": (SAMPLE_RATE_MHZ,),
        "Modulation": (MASK_LINES,),
        "Status Lines": STATUS_LINES,
        "Static Lines": STATIC_LINES,
    }
    # What sets the shape of each image.
    fixed = [*SCAN_GEOMETRY_INPUTS, SAMPLE_RATE_MHZ]
    saves = [MASKS]
    emits = {"intensity": ImageOutput}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        raster = PortRaster.for_run(ctx)
        hold_static_lines(
            raster.daq.config.device_name, tuple(ctx.inputs[each] for each in STATIC_LINES)
        )
        for _ in count_off(ctx, ctx.inputs[NUM_FRAMES], "frame"):
            ctx.output("intensity", ImageOutput).add(raster.frame(), raster.channel_labels)
