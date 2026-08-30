"""Split confocal: the same scan, each pixel's samples split into two windows.

The ``run()`` body duplicates confocal's shape, and that is correct. Ten lines
of duplicated orchestration is cheaper than a template method with a mode flag,
which is the trap ``BaseModality`` fell into. Sharing happens in ``operations/``,
where the code is genuinely identical.

The raw pixel stream is a declared output. In v3.0 it travelled as
``_pending_auxiliary["raw_pixel_stream"]``, was picked up by
``append_auxiliary_payload``, buffered in memory and written to an npz side
channel no display ever saw -- a second output smuggled through storage because
there was no way to declare one.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.modulation import load_mask
from pyrpoc.core.params import (
    DaqGroup,
    ModulationGroup,
    SaveGroup,
    ScanGroup,
    SplitGroup,
    group,
    int_field,
)
from pyrpoc.core.streams import Image2D, Samples4D
from pyrpoc.devices import DAQ, Galvo
from pyrpoc.operations.modulation import split_mask_ttl
from pyrpoc.operations.raster import pixel_samples
from pyrpoc.operations.split_raster import split_raster_scan
from pyrpoc.run.program import Program

from .registry import program_registry


@dataclass
class SplitConfocalParams:
    scan: ScanGroup = group(ScanGroup, "Scan")
    daq: DaqGroup = group(DaqGroup, "DAQ")
    split: SplitGroup = group(SplitGroup, "Split")
    modulation: ModulationGroup = group(ModulationGroup, "Modulation")
    save: SaveGroup = group(SaveGroup, "Save")
    num_frames: int = int_field(
        "Frames", 1, minimum=1, tooltip="Number of frames to capture"
    )


def channel_labels(daq: DAQ) -> list[str]:
    """``ai0_t0``, ``ai0_t2``, ``ai1_t0``, ... -- interleaved, as in v3.0."""
    labels: list[str] = []
    for index in daq.config.ai_channels:
        labels.append(f"ai{index}_t0")
        labels.append(f"ai{index}_t2")
    return labels


def build_ttl(params: SplitConfocalParams, daq: DAQ) -> dict:
    """Mask TTL gated to the first ``t0_samples`` of every pixel."""
    if not params.modulation.masks:
        return {}
    loaded = [(binding, load_mask(binding.path)) for binding in params.modulation.masks]
    return split_mask_ttl(
        loaded,
        scan=params.scan,
        pixel_samples=pixel_samples(params.scan.dwell_time_us, params.daq.sample_rate_hz),
        device_name=daq.config.device_name,
        t0_samples=params.split.t0_samples,
    )


@program_registry.register("split_confocal")
class SplitConfocal(Program):
    uses = [Galvo, DAQ]
    params = SplitConfocalParams
    emits = {"intensity": Image2D, "raw_pixel_stream": Samples4D}

    def run(self, ctx) -> None:
        p: SplitConfocalParams = ctx.params
        daq: DAQ = ctx.devices[DAQ]
        galvo: Galvo = ctx.devices[Galvo]

        ttl = build_ttl(p, daq)
        labels = channel_labels(daq)
        total = "" if ctx.continuous else f"/{p.num_frames}"

        for index in ctx.frames(p.num_frames):
            ctx.status(f"frame {index + 1}{total}")
            split, raw = split_raster_scan(
                **p.scan,
                **p.daq,
                **p.split,
                **daq.config,
                **galvo.config,
                ttl=ttl,
            )
            ctx.publish("intensity", split, channels=labels)
            ctx.publish("raw_pixel_stream", raw)
