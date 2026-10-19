"""Parking the galvos at one point, typed or picked off an image."""

from __future__ import annotations

import nidaqmx as nx
import nidaqmx.errors

from pyrpoc.plugins.devices import DAQ, DaqError, Galvo
from pyrpoc.structs.plugins.params import BlockMap
from pyrpoc.structs.plugins.programs.picks import Pick, PixelPick

from ..parameter_groups import Point, PointGroup
from .daq_tasks import add_galvo_channels


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
