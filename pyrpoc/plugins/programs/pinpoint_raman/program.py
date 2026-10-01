"""Pinpoint Raman: park the galvos at one point and take a spectrum there.

The point comes from ``PointGroup``, typed or set by an ``ArmAndRun`` runner
from a clicked pixel; ``run`` does not know a display exists. The parking is
real and is left in place when the run ends, so the beam stays on the spot for
a spectrometer read by external software.
"""

from __future__ import annotations

import nidaqmx as nx
import nidaqmx.errors

from pyrpoc.plugins.devices import DAQ, DaqError, Galvo
from pyrpoc.structs.data_library.data import Spectrum1D
from pyrpoc.structs.plugins.params import BlockMap
from pyrpoc.structs.plugins.programs.picks import Pick, PixelPick
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.functions.daq_tasks import add_galvo_channels
from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.parameter_groups import Point, PointGroup, SpectrumGroup
from ..building_blocks.runners import ArmAndRun, Continuous, Single
from .spectrum import synthetic_spectrum


def park_galvos(daq: DAQ, galvo: Galvo, point: Point) -> None:
    """Move the galvos to ``point`` and leave them there.

    An un-clocked task, so the write is one on-demand sample per channel.
    Closing the task frees the AO channels for the next scan; the card holds
    its last written voltage, so the mirrors stay parked after the run ends.
    """
    try:
        with nx.Task() as task:
            add_galvo_channels(task, daq.config.device_name, galvo)
            # One sample per channel, so write() auto-starts.
            task.write([point.fast_v, point.slow_v])
    except nidaqmx.errors.Error as exc:
        raise DaqError(f"could not park the galvos: {exc}") from exc


def aim_at_pick(pick: Pick, params: BlockMap) -> None:
    """Point the galvos at the picked pixel: its volts become the target."""
    # Narrows for the type checker: the runner asks for a PixelPick.
    if not isinstance(pick, PixelPick):
        raise TypeError(f"expected a PixelPick, got {type(pick).__name__}")
    params[PointGroup].target = Point.from_pixel(pick.dataset, pick.x, pick.y)


@program_registry.register("pinpoint_raman")
class PinpointRaman(Program):
    display_name = "Galvo pinpoint"
    group = "Spectrum"
    order = 30
    uses = [Galvo]
    params = [PointGroup, SpectrumGroup]
    emits = {"spectrum": Spectrum1D}
    runners = [
        Single(),
        Continuous(),
        ArmAndRun(
            PixelPick,
            aim_at_pick,
            label="Acquire at point…",
            icon=None,
            tooltip="Arm, then provide what this run needs to start it",
        ),
    ]

    def run(self, ctx: RunContext) -> None:
        point = ctx.params[PointGroup].target
        spec = ctx.params[SpectrumGroup]

        ctx.status(f"parking at {point.fast_v:.3f} / {point.slow_v:.3f} V")
        # Both devices are bound though ``uses`` names only the galvo: claims
        # follow ``backed_by``, and the mirrors are voltages on the card.
        park_galvos(ctx.devices[DAQ], ctx.devices[Galvo], point)

        for index in count_off(ctx, spec.num_frames, "spectrum"):
            ctx.sleep(spec.integration_ms / 1000.0)
            ctx.publish(
                "spectrum", synthetic_spectrum(spec, point, frame_index=index), channels=["raman"]
            )
