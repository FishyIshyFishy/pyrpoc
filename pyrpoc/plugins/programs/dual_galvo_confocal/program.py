"""Dual galvo confocal: confocal's scan on the first galvo pair while the second
pair plays its own signal on the same AO clock, to test four AO channels at once."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pyrpoc.plugins.devices import DAQ, DualGalvo
from pyrpoc.structs.data_library.data import Image2D
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.functions.galvo_raster import Raster
from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.parameter_groups import (
    DaqGroup,
    FrameCountGroup,
    ModulationGroup,
    ScanGroup,
    SecondGalvosGroup,
)
from ..building_blocks.runners import Continuous, Single


def second_waveform(
    second: SecondGalvosGroup, scan_waveform: np.ndarray, sample_rate_hz: float
) -> np.ndarray:
    """The second pair's ``(fast, slow)`` rows, as long as the scan's."""
    if second.pattern == "raster":
        return scan_waveform.copy()
    phase = 2.0 * np.pi * second.frequency_hz * np.arange(scan_waveform.shape[1]) / sample_rate_hz
    return second.amplitude * np.vstack((np.sin(phase), np.cos(phase))) + second.offset


# The second pair's lines take this many times longer than the scan's: an
# octave lower in pitch at 2, and still fast enough to hear and see.
SECOND_PAIR_SLOWDOWN = 2


def slowed_raster(scan: ScanGroup, scan_waveform: np.ndarray, samples_per_line: int) -> np.ndarray:
    """The scan's raster with each line ``SECOND_PAIR_SLOWDOWN`` times longer and
    proportionally fewer lines, so it still sweeps both axes once per frame."""
    total = scan_waveform.shape[1]
    line = np.repeat(scan_waveform[0, :samples_per_line], SECOND_PAIR_SLOWDOWN)
    lines = -(-total // len(line))
    slow_axis = (
        np.linspace(-1.0, 1.0, lines, endpoint=False) * scan.slow_axis_amplitude
        + scan.slow_axis_offset
    )
    fast_raster = np.resize(line, total)
    slow_raster = np.repeat(slow_axis, len(line))[:total]
    return np.vstack((fast_raster, slow_raster))


@program_registry.register("dual_galvo_confocal")
class DualGalvoConfocal(Program):
    display_name = "Dual galvo (4 AO)"
    group = "Confocal Fluorescence"
    order = 12
    uses = [DualGalvo, DAQ]
    params = [ScanGroup, FrameCountGroup, DaqGroup, SecondGalvosGroup, ModulationGroup]
    emits = {"intensity": Image2D}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        galvos = ctx.devices[DualGalvo]
        raster = Raster.on_channels(ctx, galvos.scan_channels)
        # Debug: the second pair plays a slowed copy of the scan, ignoring SecondGalvosGroup.
        scan = raster.scan
        second = slowed_raster(scan, raster.waveform, scan.total_x * raster.samples_per_pixel)
        print(np.shape(galvos.scan_channels + galvos.second_channels))
        print(raster)
        raster = replace(
            raster,
            ao_channels=galvos.scan_channels + galvos.second_channels,
            waveform=np.vstack((raster.waveform, second)),
        )
        for _ in count_off(ctx, ctx.params[FrameCountGroup].num_frames, "frame"):
            ctx.publish("intensity", raster.frame(), channels=raster.channel_labels)
