"""Split confocal: the same raster, read as two time-gated windows per pixel.

Same scan and mask as confocal; the only differences are the two extra timing
parameters, the t0-gated mask window, and the t0/t2 readout — all confined here.
"""

from __future__ import annotations

import numpy as np

from pyrpoc_next.acquisition.modalities.base import modality_registry
from pyrpoc_next.acquisition.modalities.scanner import ScannerModality, scanner_parameter_groups
from pyrpoc_next.structs.keys import InstrumentKey, ModalityKey, ModifierKey
from pyrpoc_next.structs.manifest import ModalityManifest
from pyrpoc_next.structs.parameters import NumberParameter
from pyrpoc_next.structs.parcels import ImageFrameParcel


def split_parameter_groups():
    groups = scanner_parameter_groups()
    groups["acquisition"] = [
        NumberParameter(label="t0 Samples", default=1, minimum=1, number_type=int),
        NumberParameter(label="t1 Samples", default=0, minimum=0, number_type=int),
        *groups["acquisition"],
    ]
    return groups


@modality_registry.register
class SplitConfocalModality(ScannerModality):
    """Confocal with per-pixel time gating into a t0 and a t2 window."""

    key = ModalityKey.split_confocal
    manifest = ModalityManifest(
        key=ModalityKey.split_confocal,
        display_name="Split Confocal",
        emitted_parcels=(ImageFrameParcel,),
        required_instruments=(InstrumentKey.ni_daq,),
        realizable_modifiers=(ModifierKey.mask,),
        parameter_groups=split_parameter_groups(),
    )

    def acquire_frame(self, index: int) -> list[ImageFrameParcel]:
        scan = self.build_scan(self.values)
        t0 = self.values["t0 Samples"]
        t1 = self.values["t1 Samples"]
        ttl = self.mask_ttl(scan, active_samples=t0)
        cube = self.daq().run(scan, ttl)

        channels, height, width, samples = cube.shape
        first = cube[:, :, :, :t0].mean(axis=3)
        gap_end = t0 + t1
        if gap_end < samples:
            second = cube[:, :, :, gap_end:].mean(axis=3)
        else:
            second = np.zeros_like(first)

        frame = np.stack((first, second), axis=1).reshape(channels * 2, height, width)
        labels = [f"ai{channel}_{window}" for channel in scan.ai_channels for window in ("t0", "t2")]
        return [ImageFrameParcel(data=frame.astype(np.float32, copy=False), channel_labels=labels)]
