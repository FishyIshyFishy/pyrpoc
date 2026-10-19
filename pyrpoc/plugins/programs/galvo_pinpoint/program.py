"""Galvo pinpoint: park the galvos at one point and leave them there.

For reading the spectrum in external software such as Andor SOLIS, so pyrpoc
does not touch the CCD. The point is typed or set by an ``ArmAndRun`` runner
from a clicked pixel, and the run records it in its parameters.
"""

from __future__ import annotations

from pyrpoc.plugins.devices import DAQ, Galvo
from pyrpoc.structs.plugins.programs.picks import PixelPick
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.functions.galvo_park import aim_at_pick, park_galvos
from ..building_blocks.parameter_groups import PointGroup
from ..building_blocks.runners import ArmAndRun, Single


@program_registry.register("galvo_pinpoint")
class GalvoPinpoint(Program):
    display_name = "Pinpoint"
    group = "Raman"
    order = 30
    uses = [Galvo]
    params = [PointGroup]
    runners = [
        Single(),
        ArmAndRun(
            PixelPick,
            aim_at_pick,
            label="Park at point…",
            icon=None,
            tooltip="Arm, then click a pixel to park the beam there",
        ),
    ]

    def run(self, ctx: RunContext) -> None:
        point = ctx.params[PointGroup].target
        ctx.status(f"parking at {point.fast_v:.3f} / {point.slow_v:.3f} V")
        # The DAQ is bound though ``uses`` does not name it: claims follow
        # ``backed_by``, and the mirrors are voltages on the card.
        park_galvos(ctx.devices[DAQ], ctx.devices[Galvo], point)
