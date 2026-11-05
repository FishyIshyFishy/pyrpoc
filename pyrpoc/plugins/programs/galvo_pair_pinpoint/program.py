"""Pinpoint: park one galvo of a pair at a spot picked off an image taken with the other, or
typed, and correct the registration from where its spot shows up."""

from __future__ import annotations

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.plugins.devices import DAQ
from pyrpoc.plugins.program_io.inputs import (
    GALVO_TARGET,
    PINPOINT_GALVO,
    REGISTRATION_INPUTS,
    Registration,
)

from ..building_blocks.functions.galvo_pair import GALVO_PAIR, chosen_galvo, correct_registration
from ..building_blocks.functions.galvo_park import aim_at_pixel, park_galvo
from ..building_blocks.runners import ArmAndRun, Single


@program_registry.register("galvo_pair_pinpoint")
class GalvoPairPinpoint(Program):
    display_name = "Pinpoint"
    group = "Karsten"
    order = 310
    devices = [DAQ]
    chooses = GALVO_PAIR
    inputs = [PINPOINT_GALVO, GALVO_TARGET, *REGISTRATION_INPUTS]
    cards = {
        "Pinpoint": (PINPOINT_GALVO, GALVO_TARGET),
        "Registration": REGISTRATION_INPUTS,
    }
    runners = [
        Single(),
        ArmAndRun(
            aim_at_pixel,
            label="Park at click…",
            icon=None,
            tooltip="Arm, then click a pixel on an image taken with the other galvo to park there",
        ),
        ArmAndRun(
            correct_registration,
            label="Correct registration…",
            icon=None,
            tooltip=(
                "Arm, then click where the parked spot shows up on an image taken with the other "
                "galvo: the offset shifts by the miss and the galvo parks again"
            ),
        ),
    ]

    def run(self, ctx: RunContext) -> None:
        parked = ctx.inputs[PINPOINT_GALVO]
        galvo = chosen_galvo(ctx, parked)
        target = ctx.inputs[GALVO_TARGET]
        fast_offset, slow_offset = Registration.from_inputs(ctx.inputs).offset_of(parked)
        fast_v, slow_v = target.fast_v + fast_offset, target.slow_v + slow_offset
        ctx.status(f"parking the {parked.lower()} galvo at {fast_v:.3f} / {slow_v:.3f} V")
        park_galvo(ctx.devices[DAQ], galvo, fast_v, slow_v)
