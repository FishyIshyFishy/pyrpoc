"""Parking a galvo at one point, typed or picked off an image."""

from __future__ import annotations

import nidaqmx as nx
import nidaqmx.errors

from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.plugins.devices import DAQ, DaqError, Galvo
from pyrpoc.plugins.program_io.inputs import GALVO_TARGET, PixelPick, Point

from .daq_tasks import add_ao_channels, galvo_channels


def park_galvo(daq: DAQ, galvo: Galvo, fast_v: float, slow_v: float) -> None:
    """Move the galvo to ``(fast_v, slow_v)`` and leave it there."""
    # An un-clocked task, so the write is one on-demand sample per channel. Closing the task frees
    # the AO channels for the next scan; the card holds its last written voltage, so the mirrors
    # stay parked after the run ends.
    try:
        with nx.Task() as task:
            add_ao_channels(task, daq.config.device_name, galvo_channels(galvo))
            # One sample per channel, so write() auto-starts.
            task.write([fast_v, slow_v])
    except nidaqmx.errors.Error as exc:
        raise DaqError(f"could not park the galvo: {exc}") from exc


def aim_at_pixel(pick: PixelPick, inputs: ProgramInputs) -> None:
    """Point the galvos at the picked pixel: its volts become the target."""
    inputs.set(GALVO_TARGET.key, Point.from_pick(pick))
