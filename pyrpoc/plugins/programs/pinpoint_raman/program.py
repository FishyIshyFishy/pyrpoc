"""Pinpoint Raman: park the galvos at one point and read a spectrum there.

The point comes from ``PointGroup``, typed or set by an ``ArmAndRun`` runner
from a clicked pixel; ``run`` does not know a display exists. The parking is
left in place when the run ends, so a continuous run keeps reading the same spot.
"""

from __future__ import annotations

from pyrpoc.plugins.devices import DAQ, AndorCCD, Galvo
from pyrpoc.structs.data_library.data import Spectrum1D
from pyrpoc.structs.plugins.programs.picks import PixelPick
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.functions.ccd_read import publish_spectrum
from ..building_blocks.functions.galvo_park import aim_at_pick, park_galvos
from ..building_blocks.parameter_groups import CcdReadoutGroup, PointGroup
from ..building_blocks.runners import ArmAndRun, Continuous, Single


@program_registry.register("pinpoint_raman")
class PinpointRaman(Program):
    display_name = "Pinpoint + spectral readout"
    group = "Raman"
    order = 20
    uses = [Galvo, AndorCCD]
    params = [PointGroup, CcdReadoutGroup]
    emits = {"spectrum": Spectrum1D}
    runners = [
        Single(),
        Continuous(),
        ArmAndRun(
            PixelPick,
            aim_at_pick,
            label="Acquire at point…",
            icon=None,
            tooltip="Arm, then click a pixel to read a spectrum there",
        ),
    ]

    def run(self, ctx: RunContext) -> None:
        point = ctx.params[PointGroup].target
        readout = ctx.params[CcdReadoutGroup].to_readout()

        ctx.status(f"parking at {point.fast_v:.3f} / {point.slow_v:.3f} V")
        # The DAQ is bound though ``uses`` does not name it: claims follow
        # ``backed_by``, and the mirrors are voltages on the card.
        park_galvos(ctx.devices[DAQ], ctx.devices[Galvo], point)
        publish_spectrum(ctx, ctx.devices[AndorCCD], readout, "spectrum")
