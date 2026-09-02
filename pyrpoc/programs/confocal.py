"""Confocal: raster the galvo, read the analog inputs, publish a frame.

The program owns its loop. There is no base class supplying
``while not should_stop``, no saving (the runner reads ``emits`` plus the Save
group and creates datasets with a save policy before ``run`` is called), and no
frame counting.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.modulation import load_mask
from pyrpoc.core.params import (
    DaqGroup,
    ModulationGroup,
    SaveGroup,
    ScanGroup,
    group,
    int_field,
)
from pyrpoc.core.streams import Image2D
from pyrpoc.data.dataset import Dataset  # noqa: F401  (documents what publish writes into)
from pyrpoc.devices import DAQ, Galvo
from pyrpoc.operations.modulation import mask_ttl
from pyrpoc.operations.raster import pixel_samples, raster_scan
from pyrpoc.run.program import Program

from .registry import program_registry


@dataclass
class ConfocalParams:
    scan: ScanGroup = group(ScanGroup, "Scan")
    daq: DaqGroup = group(DaqGroup, "DAQ")
    modulation: ModulationGroup = group(ModulationGroup, "Modulation")
    save: SaveGroup = group(SaveGroup, "Save")
    num_frames: int = int_field(
        "Frames", 1, minimum=1, tooltip="Number of frames to capture"
    )


def channel_labels(daq: DAQ) -> list[str]:
    return [f"ai{index}" for index in daq.config.ai_channels]


def build_ttl(params: ConfocalParams, daq: DAQ) -> dict:
    """Load the bound masks and turn them into per-pixel TTL waveforms.

    Done once before the loop rather than once per frame, and here rather than
    inside the operation because operations/ may not read files.
    """
    if not params.modulation.masks:
        return {}
    loaded = [(binding, load_mask(binding.path)) for binding in params.modulation.masks]
    return mask_ttl(
        loaded,
        scan=params.scan,
        pixel_samples=pixel_samples(params.scan.dwell_time_us, params.daq.sample_rate_hz),
        device_name=daq.config.device_name,
    )


@program_registry.register("confocal")
class Confocal(Program):
    uses = [Galvo, DAQ]
    params = ConfocalParams
    emits = {"intensity": Image2D}

    def run(self, ctx) -> None:
        p: ConfocalParams = ctx.params
        daq: DAQ = ctx.devices[DAQ]
        galvo: Galvo = ctx.devices[Galvo]

        ttl = build_ttl(p, daq)
        labels = channel_labels(daq)
        total = "" if ctx.continuous else f"/{p.num_frames}"

        for index in ctx.frames(p.num_frames):
            ctx.status(f"frame {index + 1}{total}")
            frame = raster_scan(
                **p.scan,
                **p.daq,
                **daq.config,
                **galvo.config,
                ttl=ttl,
            )
            ctx.publish("intensity", frame, channels=labels)
