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
        second = second_waveform(
            ctx.params[SecondGalvosGroup], raster.waveform, raster.sample_rate_hz
        )
        raster = replace(
            raster,
            ao_channels=galvos.scan_channels + galvos.second_channels,
            waveform=np.vstack((raster.waveform, second)),
        )
        for _ in count_off(ctx, ctx.params[FrameCountGroup].num_frames, "frame"):
            ctx.publish("intensity", raster.frame(), channels=raster.channel_labels)
