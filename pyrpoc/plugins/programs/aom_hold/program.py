"""AOM hold: hold digital lines high until the run is stopped, so the AOMs
they gate stay open without a scan, for alignment.

Stopping the run brings the lines low again. Holding them is a run so that the
DAQ stays claimed: no scan can start and write the lines from under it.
"""

from __future__ import annotations

from pyrpoc.plugins.devices import DAQ
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.functions.hold_lines import hold_high
from ..building_blocks.parameter_groups import HeldLineGroup
from ..building_blocks.runners import Single


@program_registry.register("aom_hold")
class AomHold(Program):
    display_name = "AOM hold (alignment)"
    group = "Custom"
    order = 60
    uses = [DAQ]
    params = [HeldLineGroup]
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        hold_high(ctx, ctx.params[HeldLineGroup].channels(ctx.devices[DAQ].config.device_name))
