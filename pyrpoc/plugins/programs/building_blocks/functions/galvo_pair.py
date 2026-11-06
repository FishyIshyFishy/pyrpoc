"""Two galvos on one sample: the names programs choose them under, which of them a run moves,
and correcting the registration from where a parked spot shows up on an image."""

from __future__ import annotations

from pyrpoc.core.device import Device
from pyrpoc.core.program import RunContext
from pyrpoc.core.program_input import ParameterError, ProgramInputs
from pyrpoc.plugins.devices import Galvo
from pyrpoc.plugins.program_io.inputs import (
    BOTH,
    GALVO_TARGET,
    PINPOINT_GALVO,
    REFERENCE,
    SCAN_GALVOS,
    SECOND,
    SECOND_OFFSET_FAST_V,
    SECOND_OFFSET_SLOW_V,
    PixelPick,
    Point,
    Registration,
)

from .galvo_raster import Drive

REFERENCE_GALVO = "Reference galvo"
SECOND_GALVO = "Second galvo"

# What a program choosing a pair declares as ``chooses``.
GALVO_PAIR: dict[str, type[Device]] = {REFERENCE_GALVO: Galvo, SECOND_GALVO: Galvo}

# The name each galvo of SCAN_GALVOS and PINPOINT_GALVO is chosen under.
CHOSEN_AS = {REFERENCE: REFERENCE_GALVO, SECOND: SECOND_GALVO}


def chosen_galvo(ctx: RunContext, galvo: str) -> Galvo:
    """The device chosen as ``galvo`` (``REFERENCE`` or ``SECOND``), once the pair is two."""
    if ctx.chosen(REFERENCE_GALVO, Galvo) is ctx.chosen(SECOND_GALVO, Galvo):
        raise ParameterError("the reference and second galvo are one device; choose two")
    return ctx.chosen(CHOSEN_AS[galvo], Galvo)


def scan_drives(ctx: RunContext) -> list[Drive]:
    """Each galvo SCAN_GALVOS names, shifted so they all cover the same field."""
    registration = Registration.from_inputs(ctx.inputs)
    scanned = ctx.inputs[SCAN_GALVOS]
    galvos = [REFERENCE, SECOND] if scanned == BOTH else [scanned]
    return [Drive(chosen_galvo(ctx, galvo), *registration.offset_of(galvo)) for galvo in galvos]


def correct_registration(pick: PixelPick, inputs: ProgramInputs) -> None:
    """Shift the second galvo's offset by how far the parked galvo's spot, picked on an image
    taken with the other, landed from its target."""
    parked = inputs.get(PINPOINT_GALVO.key)
    scanned = pick.parameters.get(SCAN_GALVOS.key)
    if scanned is None:
        raise ParameterError(f"{pick.source_label} does not say which galvo scanned it")
    if scanned in (parked, BOTH):
        raise ParameterError("pick the spot on an image scanned with only the other galvo")
    target: Point = inputs.get(GALVO_TARGET.key)
    landed = Point.from_pick(pick)
    # The offset is off by the miss: short of it when the second galvo was parked, and over it
    # when the reference was, since then the second galvo's scan is what misplaced the spot.
    sign = 1.0 if parked == SECOND else -1.0
    fast_miss = target.fast_v - landed.fast_v
    slow_miss = target.slow_v - landed.slow_v
    inputs.set(SECOND_OFFSET_FAST_V.key, inputs.get(SECOND_OFFSET_FAST_V.key) + sign * fast_miss)
    inputs.set(SECOND_OFFSET_SLOW_V.key, inputs.get(SECOND_OFFSET_SLOW_V.key) + sign * slow_miss)
