"""Pinpoint Raman: park the galvos at one point, typed or picked off an image,
and read a spectrum there."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices import DAQ, AndorCCD, Galvo
from pyrpoc.plugins.program_io.inputs import CCD_EXPOSURE_S, CCD_FRAMES_TO_AVERAGE, GALVO_TARGET
from pyrpoc.plugins.program_io.outputs import Spectrum
from pyrpoc.plugins.programs.building_blocks.functions.ccd_read import spectrum_readout

from ..building_blocks.functions.ccd_read import publish_spectrum
from ..building_blocks.functions.galvo_park import aim_at_pixel, park_galvos
from ..building_blocks.runners import ArmAndRun, Continuous, Single


@program_registry.register("pinpoint_raman")
class PinpointRaman(Program):
    display_name = "Pinpoint + spectral readout"
    group = "Raman"
    order = 20
    devices = [Galvo, AndorCCD]
    sections = [
        InputSection("Point", (GALVO_TARGET,)),
        InputSection("CCD Readout", (CCD_EXPOSURE_S, CCD_FRAMES_TO_AVERAGE)),
    ]
    emits = {"spectrum": Spectrum}
    runners = [
        Single(),
        Continuous(),
        ArmAndRun(
            aim_at_pixel,
            label="Acquire at point…",
            icon=None,
            tooltip="Arm, then click a pixel to read a spectrum there",
        ),
    ]

    def run(self, ctx: RunContext) -> None:
        point = ctx.inputs[GALVO_TARGET]
        readout = spectrum_readout(ctx.inputs)

        ctx.status(f"parking at {point.fast_v:.3f} / {point.slow_v:.3f} V")
        # The DAQ is bound though ``devices`` does not name it: claims follow
        # ``backed_by``, and the mirrors are voltages on the card.
        park_galvos(ctx.devices[DAQ], ctx.devices[Galvo], point)
        publish_spectrum(ctx, ctx.devices[AndorCCD], readout, "spectrum")
