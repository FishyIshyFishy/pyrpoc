"""Holding digital lines high until the run is stopped, such as AOM gates
while aligning: the loop itself, and the stand-in a scanning program's
"AOMs on" toggle runs in place of its scan."""

from __future__ import annotations

import nidaqmx.errors

from pyrpoc.plugins.devices import DAQ, DaqError
from pyrpoc.structs.plugins.params import ParameterError
from pyrpoc.structs.plugins.programs.program import Program, RunContext

from ..parameter_groups.modulation import ModulationGroup
from .daq_tasks import held_lines

# How long each wait between checks for Stop is; Stop ends one at once.
WAIT_S = 60.0


def hold_high(ctx: RunContext, channels: list[str]) -> None:
    """Hold ``channels`` high until the run is stopped, then low again. Low on
    any error too, since the lines drop as the hold's block exits."""
    named = ", ".join(channels)
    try:
        with held_lines(dict.fromkeys(channels, True)):
            ctx.status(f"{named} held high; Stop to release")
            while True:
                ctx.sleep(WAIT_S)
    except nidaqmx.errors.Error as exc:
        raise DaqError(f"could not hold {named}: {exc}") from exc


class HoldMaskLines(Program):
    """Every line a mask in Modulation drives, held high in place of a scan.

    Not registered, so not listed: the ``Hold`` runner starts it from a
    scanning program, whose Modulation table already names the AOM lines. It
    claims the DAQ, so no scan can start while the lines are held.
    """

    display_name = "AOMs on"
    uses = [DAQ]
    params = [ModulationGroup]

    def run(self, ctx: RunContext) -> None:
        device_name = ctx.devices[DAQ].config.device_name
        masks = ctx.params[ModulationGroup].masks
        # A dict, so a line two masks share is held once, in table order.
        channels = list(dict.fromkeys(mask.channel(device_name) for mask in masks))
        if not channels:
            raise ParameterError("bind a mask in Modulation to name the lines to hold")
        hold_high(ctx, channels)
