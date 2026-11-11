"""Hyperspectral: a confocal frame at each delay of a sweep of the Zaber delay stage, then the
stage back where it started."""

from __future__ import annotations

from pyrpoc.core.program import Cancelled, Program, RunContext, program_registry
from pyrpoc.plugins.devices import DAQ, Galvo, ZaberStage
from pyrpoc.plugins.program_io.inputs import (
    DELAY_START_UM,
    DELAY_STEPS,
    DELAY_STOP_UM,
    MASK_LINES,
    MASKS,
    SAMPLE_RATE_MHZ,
    SCAN_GEOMETRY_INPUTS,
    DelaySweep,
)
from pyrpoc.plugins.program_io.outputs import HyperspectralOutput

from ..building_blocks.functions.galvo_raster import Raster
from ..building_blocks.runners import Single

DELAY_SWEEP_INPUTS = (DELAY_START_UM, DELAY_STOP_UM, DELAY_STEPS)


def move_delay(ctx: RunContext, stage: ZaberStage, position_um: float) -> None:
    """Move and wait for the stage to stop; a stopped run interrupts the wait."""
    stage.move_to(position_um)
    stage.wait_until_stopped(ctx.sleep)


def home_if_needed(ctx: RunContext, stage: ZaberStage) -> None:
    """Home a stage that has not been, since absolute positions count from home."""
    if stage.homed():
        return
    ctx.status("homing the delay stage")
    stage.home()
    stage.wait_until_stopped(ctx.sleep)


def sweep(ctx: RunContext, stage: ZaberStage, raster: Raster, delays: tuple[float, ...]) -> None:
    """Image at every delay, adding each image as it is taken."""
    cube = ctx.output("hyperspectral", HyperspectralOutput)
    for index, delay_um in enumerate(delays):
        ctx.check_cancel()
        ctx.status(f"delay {index + 1}/{len(delays)} at {delay_um:.1f} µm")
        move_delay(ctx, stage, delay_um)
        cube.add_at(delay_um, raster.frame(), raster.channel_labels)


@program_registry.register("hyperspectral")
class Hyperspectral(Program):
    display_name = "Hyperspectral"
    group = "Raman"
    order = 40
    devices = [Galvo, DAQ, ZaberStage]
    inputs = [
        *SCAN_GEOMETRY_INPUTS,
        SAMPLE_RATE_MHZ,
        MASKS,
        MASK_LINES,
        *DELAY_SWEEP_INPUTS,
    ]
    # MASKS has no row: masks are drawn and edited in a Mask Editor.
    cards = {
        "Delay Sweep": DELAY_SWEEP_INPUTS,
        "Scan": SCAN_GEOMETRY_INPUTS,
        "DAQ": (SAMPLE_RATE_MHZ,),
        "Modulation": (MASK_LINES,),
    }
    emits = {"hyperspectral": HyperspectralOutput}
    # Once only: a sweep is one pass over the delays, never repeated or looped.
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        stage = ctx.devices[ZaberStage]
        delays = DelaySweep.from_inputs(ctx.inputs).positions_um()
        # Built first, so a scan that cannot be set up fails before the stage moves.
        raster = Raster.for_run(ctx)
        try:
            home_if_needed(ctx, stage)
            origin = stage.position_um()
            sweep(ctx, stage, raster, delays)
        except Cancelled:
            stage.stop()
            raise
        # Back where it started, so imaging after the sweep keeps the delay it had.
        ctx.status("returning the delay stage to where it started")
        move_delay(ctx, stage, origin)
