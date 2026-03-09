from __future__ import annotations

from typing import Any

import numpy as np

from pyrpoc.backend_utils.contracts import Parameter
from pyrpoc.backend_utils.data import DataImage
from pyrpoc.instruments.confocal_daq import ConfocalDAQInstrument
from pyrpoc.optocontrols.mask import MaskOptoControl
from .base_modality import BaseModality
from .mod_registry import modality_registry


@modality_registry.register("sim_confocal")
class SimConfocalModality(BaseModality):
    MODALITY_KEY = "sim_confocal"
    DISPLAY_NAME = "Simulated Confocal"
    PARAMETERS = {
        "scan": [
            Parameter(
                label="X Pixels",
                param_type=int,
                default=256,
                minimum=8,
                tooltip="Number of pixels in X",
            ),
            Parameter(
                label="Y Pixels",
                param_type=int,
                default=256,
                minimum=8,
                tooltip="Number of pixels in Y",
            ),
            Parameter(
                label="Dwell Time (us)",
                param_type=float,
                default=10.0,
                minimum=0.1,
                tooltip="Pixel dwell time for simulated scan",
            ),
            Parameter(
                label="Frames",
                param_type=int,
                default=1,
                minimum=1,
                tooltip="Number of frames per acquisition cycle",
            ),
        ],
        "save": [
            Parameter(
                label='Save Path',
                param_type = str,
                default = None,
            ),
        ]
    }
    REQUIRED_INSTRUMENTS = [ConfocalDAQInstrument]
    OPTIONAL_INSTRUMENTS = []
    ALLOWED_OPTOCONTROLS = [MaskOptoControl]
    OUTPUT_DATA_TYPE = DataImage
    ALLOWED_DISPLAYS = ["sim_image"]

    def __init__(self):
        super().__init__()
        self._frame_idx = 0

    def configure(
        self,
        params: dict[str, Any],
        instruments: dict[type[ConfocalDAQInstrument], ConfocalDAQInstrument],
    ) -> None:
        self._params = dict(params)
        self._instruments = dict(instruments)
        self._configured = True

    def start(self) -> None:
        if not self._configured:
            raise RuntimeError("modality must be configured before start")
        self._running = True

    def acquire_once(self) -> DataImage:
        if not self._running:
            raise RuntimeError("modality is not running")

        x_pixels = int(self._params["X Pixels"])
        y_pixels = int(self._params["Y Pixels"])

        x = np.linspace(0, 1, x_pixels, dtype=float)
        y = np.linspace(0, 1, y_pixels, dtype=float)
        xx, yy = np.meshgrid(x, y)
        image = (np.sin((xx + self._frame_idx * 0.03) * 12.0) + np.cos(yy * 8.0)) * 0.5
        noise = np.random.normal(loc=0.0, scale=0.05, size=(y_pixels, x_pixels))
        self._frame_idx += 1

        metadata = {
            "frame_index": self._frame_idx,
            "dwell_time_us": float(self._params["Dwell Time (us)"]),
            "frames": int(self._params["Frames"]),
        }
        return DataImage(name="sim_confocal_frame", value=image + noise, metadata=metadata)

    def stop(self) -> None:
        self._running = False


ConfocalModality = SimConfocalModality
