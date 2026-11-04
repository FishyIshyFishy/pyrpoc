"""Galvo pinpoint: park the galvos at one point, typed or picked off an image, and leave
them there."""

# For reading the spectrum in external software such as Andor SOLIS, so pyrpoc does not touch the
# CCD.

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices import DAQ, Galvo
from pyrpoc.plugins.program_io.inputs import GALVO_TARGET

from ..building_blocks.functions.galvo_park import aim_at_pixel, park_galvos
from ..building_blocks.runners import ArmAndRun, Single


@program_registry.register("galvo_pinpoint")
class GalvoPinpoint(Program):
    display_name = "Pinpoint"
    group = "Raman"
    order = 30
    devices = [Galvo]
    sections = [
        InputSection("Point", (GALVO_TARGET,)),
    ]
    runners = [
        Single(),
        ArmAndRun(
            aim_at_pixel,
            label="Park at point…",
            icon=None,
            tooltip="Arm, then click a pixel to park the beam there",
        ),
    ]

    def run(self, ctx: RunContext) -> None:
        point = ctx.inputs[GALVO_TARGET]
        ctx.status(f"parking at {point.fast_v:.3f} / {point.slow_v:.3f} V")
        # The DAQ is bound though ``devices`` does not name it: claims follow
        # ``backed_by``, and the mirrors are voltages on the card.
        park_galvos(ctx.devices[DAQ], ctx.devices[Galvo], point)
