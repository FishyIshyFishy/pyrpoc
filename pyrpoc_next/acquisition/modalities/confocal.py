"""Confocal: a galvo raster read as mean intensity per pixel."""

from __future__ import annotations

import numpy as np

from pyrpoc_next.acquisition.modalities.base import modality_registry
from pyrpoc_next.acquisition.modalities.scanner import ScannerModality, scanner_parameter_groups
from pyrpoc_next.structs.keys import InstrumentKey, ModalityKey, ModifierKey
from pyrpoc_next.structs.manifest import ModalityManifest
from pyrpoc_next.structs.parcels import ImageFrameParcel


@modality_registry.register
class ConfocalModality(ScannerModality):
    """Standard confocal intensity imaging."""

    key = ModalityKey.confocal
    manifest = ModalityManifest(
        key=ModalityKey.confocal,
        display_name="Confocal",
        emitted_parcels=(ImageFrameParcel,),
        required_instruments=(InstrumentKey.ni_daq,),
        realizable_modifiers=(ModifierKey.mask,),
        parameter_groups=scanner_parameter_groups(),
    )

    def acquire_frame(self, index: int) -> list[ImageFrameParcel]:
        scan = self.build_scan(self.values)
        ttl = self.mask_ttl(scan, active_samples=scan.pixel_samples)
        cube = self.daq().run(scan, ttl)
        frame = cube.mean(axis=3).astype(np.float32, copy=False)
        labels = [f"ai{channel}" for channel in scan.ai_channels]
        return [ImageFrameParcel(data=frame, channel_labels=labels)]
