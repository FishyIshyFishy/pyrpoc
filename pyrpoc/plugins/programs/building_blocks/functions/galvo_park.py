"""Parking the galvos at one point, typed or picked off an image."""

from __future__ import annotations

import nidaqmx as nx
import nidaqmx.errors

from pyrpoc.core.input_requests import PixelPick
from pyrpoc.core.program_inputs import BlockMap
from pyrpoc.plugins.devices import DAQ, DaqError, Galvo

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


def aim_at_pixel(pick: PixelPick, params: BlockMap) -> None:
    """Point the galvos at the picked pixel: its volts become the target."""
    params[PointGroup].target = Point.from_pick(pick)
