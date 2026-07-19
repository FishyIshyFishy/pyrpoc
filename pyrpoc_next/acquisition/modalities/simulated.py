"""Simulated modalities: run the full data path with no hardware.

Separate plugins from the real modalities — the real path never imports these.
The image simulator realizes a mask by brightening pixels (the toy analogue of a
TTL gate); the FLIM simulator emits toy histogram cubes.
"""

from __future__ import annotations

import numpy as np

from pyrpoc_next.acquisition.modalities.base import Modality, modality_registry
from pyrpoc_next.acquisition.modalities.scanner import (
    ScannerModality,
    acquisition_parameters,
    scanner_parameter_groups,
)
from pyrpoc_next.acquisition.modifiers.mask import MaskModifier
from pyrpoc_next.instruments.toy import boost_masked_pixels
from pyrpoc_next.structs.keys import InstrumentKey, ModalityKey, ModifierKey
from pyrpoc_next.structs.manifest import ModalityManifest
from pyrpoc_next.structs.parameters import NumberParameter
from pyrpoc_next.structs.parcels import HistogramCubeParcel, ImageFrameParcel


@modality_registry.register
class SimulatedModality(ScannerModality):
    """Toy intensity imaging: same shape as confocal, driven by the simulated DAQ."""

    key = ModalityKey.simulated
    manifest = ModalityManifest(
        key=ModalityKey.simulated,
        display_name="Simulated Scan",
        emitted_parcels=(ImageFrameParcel,),
        required_instruments=(InstrumentKey.simulated_daq,),
        realizable_modifiers=(ModifierKey.mask,),
        parameter_groups=scanner_parameter_groups(),
    )

    def acquire_frame(self, index: int) -> list[ImageFrameParcel]:
        scan = self.build_scan(self.values)
        cube = self.instruments[InstrumentKey.simulated_daq].run(scan)
        frame = cube.mean(axis=3).astype(np.float32, copy=False)
        for modifier in self.modifiers:
            if isinstance(modifier, MaskModifier) and modifier.mask is not None:
                boost_masked_pixels(frame, modifier.mask)
        labels = [f"ai{channel}" for channel in scan.ai_channels]
        return [ImageFrameParcel(data=frame, channel_labels=labels)]


def simulated_flim_groups():
    return {
        "scan": [
            NumberParameter(label="X Pixels", default=128, minimum=8, number_type=int),
            NumberParameter(label="Y Pixels", default=128, minimum=8, number_type=int),
        ],
        "timetagger": [
            NumberParameter(label="Histogram Bins", default=125, minimum=1, number_type=int),
            NumberParameter(label="Histogram Bin Width (ps)", default=100, minimum=1, number_type=int),
            NumberParameter(label="Laser Frequency (MHz)", default=80.0, minimum=1e-6),
        ],
        "acquisition": acquisition_parameters(),
    }


@modality_registry.register
class SimulatedFlimModality(Modality):
    """Toy lifetime imaging: emits toy histogram cubes from the simulated tagger."""

    key = ModalityKey.simulated_flim
    manifest = ModalityManifest(
        key=ModalityKey.simulated_flim,
        display_name="Simulated FLIM",
        emitted_parcels=(ImageFrameParcel, HistogramCubeParcel),
        required_instruments=(InstrumentKey.simulated_tagger,),
        realizable_modifiers=(),
        parameter_groups=simulated_flim_groups(),
    )

    def acquire_frame(self, index: int) -> list:
        values = self.values
        tagger = self.instruments[InstrumentKey.simulated_tagger]
        bin_width = values["Histogram Bin Width (ps)"]
        laser_period = round(1e6 / values["Laser Frequency (MHz)"])
        cube = tagger.flim_frame(values["X Pixels"], values["Y Pixels"], values["Histogram Bins"],
                                 bin_width, laser_period)
        intensity = cube.sum(axis=2)[np.newaxis].astype(np.float32, copy=False)
        return [
            ImageFrameParcel(data=intensity, channel_labels=["intensity"]),
            HistogramCubeParcel(data=cube, bin_width_ps=float(bin_width), laser_period_ps=float(laser_period)),
        ]
