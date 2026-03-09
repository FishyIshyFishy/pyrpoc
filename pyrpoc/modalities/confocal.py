from __future__ import annotations

from typing import Any

import numpy as np

from pyrpoc.backend_utils.array_contracts import CONTRACT_CHW_FLOAT32
from pyrpoc.backend_utils.contracts import Parameter
from pyrpoc.instruments.confocal_daq import ConfocalDAQInstrument
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from pyrpoc.optocontrols.mask import MaskOptoControl

from .acquisition_functions.daq_helpers import (
    DaqUnavailableError,
    acquire_confocal_frame_with_daq,
    extract_mask_contexts,
)
from .acquisition_functions.toy_data import generate_toy_confocal_frame
from .base_modality import BaseModality
from .mod_registry import modality_registry


@modality_registry.register("confocal")
class ConfocalModality(BaseModality):
    MODALITY_KEY = "confocal"
    DISPLAY_NAME = "Confocal"
    PARAMETERS = {
        "scan": [
            Parameter(
                label="X Pixels",
                param_type=int,
                default=512,
                minimum=8,
                tooltip="Number of pixels in X",
            ),
            Parameter(
                label="Y Pixels",
                param_type=int,
                default=512,
                minimum=8,
                tooltip="Number of pixels in Y",
            ),
            Parameter(
                label="Extra Steps Left",
                param_type=int,
                default=300,
                minimum=0,
                tooltip="Extra scan steps at left edge (stored only for now)",
            ),
            Parameter(
                label="Extra Steps Right",
                param_type=int,
                default=20,
                minimum=0,
                tooltip="Extra scan steps at right edge (stored only for now)",
            ),
            Parameter(
                label="Fast Axis Offset",
                param_type=float,
                default=0.0,
                tooltip="Fast-axis offset",
            ),
            Parameter(
                label="Fast Axis Amplitude",
                param_type=float,
                default=1.0,
                minimum=1e-6,
                tooltip="Fast-axis amplitude",
            ),
            Parameter(
                label="Slow Axis Offset",
                param_type=float,
                default=0.0,
                tooltip="Slow-axis offset",
            ),
            Parameter(
                label="Slow Axis Amplitude",
                param_type=float,
                default=1.0,
                minimum=1e-6,
                tooltip="Slow-axis amplitude",
            ),
            Parameter(
                label="Dwell Time (us)",
                param_type=float,
                default=2.0,
                minimum=0.1,
                tooltip="Pixel dwell time",
            ),
        ],
    }
    REQUIRED_INSTRUMENTS = [ConfocalDAQInstrument]
    OPTIONAL_INSTRUMENTS = []
    ALLOWED_OPTOCONTROLS = [MaskOptoControl]
    OUTPUT_DATA_CONTRACT = CONTRACT_CHW_FLOAT32
    ALLOWED_DISPLAYS = ["tiled_2d", "multichan_overlay"]

    def __init__(self):
        super().__init__()
        self._frame_idx = 0
        self._mask_contexts: list[dict[str, Any]] = []
        self._daq_instrument: ConfocalDAQInstrument | None = None

    def configure(
        self,
        params: dict[str, Any],
        instruments: dict[type[ConfocalDAQInstrument], ConfocalDAQInstrument],
        opto_controls: list[tuple[BaseOptoControl, tuple[Any, ...]]],
    ) -> None:
        self._params = dict(params)
        self._instruments = dict(instruments)
        instrument = self._instruments.get(ConfocalDAQInstrument)
        if instrument is None:
            raise RuntimeError("ConfocalDAQInstrument missing during configure")
        self._daq_instrument = instrument
        self._mask_contexts = extract_mask_contexts(opto_controls)
        self._frame_idx = 0
        self._configured = True

    def start(self) -> None:
        if not self._configured:
            raise RuntimeError("modality must be configured before start")
        if self._daq_instrument is None:
            raise RuntimeError("ConfocalDAQInstrument not configured")
        self._running = True

    def acquire_once(self) -> np.ndarray:
        if not self._running:
            raise RuntimeError("modality is not running")
        if self._daq_instrument is None:
            raise RuntimeError("ConfocalDAQInstrument unavailable")

        scan_settings = self._get_scan_settings()
        active_ai_channels = self._get_active_ai_channels()

        try:
            frame = acquire_confocal_frame_with_daq(
                daq_instrument=self._daq_instrument,
                active_ai_channels=active_ai_channels,
                mask_contexts=self._mask_contexts,
                **scan_settings,
            )
        except DaqUnavailableError:
            print('daq is unavalable')
            frame = generate_toy_confocal_frame(
                x_pixels=scan_settings["x_pixels"],
                y_pixels=scan_settings["y_pixels"],
                active_channels=active_ai_channels,
                frame_index=self._frame_idx,
                mask_contexts=self._mask_contexts,
                fast_axis_offset=scan_settings["fast_axis_offset"],
                fast_axis_amplitude=scan_settings["fast_axis_amplitude"],
                slow_axis_offset=scan_settings["slow_axis_offset"],
                slow_axis_amplitude=scan_settings["slow_axis_amplitude"],
            )
        self._frame_idx += 1
        return frame.astype(np.float32, copy=False)

    def stop(self) -> None:
        self._running = False

    def _get_active_ai_channels(self) -> list[int]:
        return [
            ai_channel
            for ai_channel, enabled in zip(
                self._daq_instrument.ai_channel_numbers if self._daq_instrument is not None else [],
                self._daq_instrument.active_ai_channels if self._daq_instrument is not None else [],
                strict=False,
            )
            if enabled
        ]

    def _get_scan_settings(self) -> dict[str, Any]:
        return {
            "x_pixels": int(self._params["X Pixels"]),
            "y_pixels": int(self._params["Y Pixels"]),
            "extra_left": int(self._params["Extra Steps Left"]),
            "extra_right": int(self._params["Extra Steps Right"]),
            "dwell_time_us": float(self._params["Dwell Time (us)"]),
            "fast_axis_offset": float(self._params["Fast Axis Offset"]),
            "fast_axis_amplitude": max(float(self._params["Fast Axis Amplitude"]), 1e-6),
            "slow_axis_offset": float(self._params["Slow Axis Offset"]),
            "slow_axis_amplitude": max(float(self._params["Slow Axis Amplitude"]), 1e-6),
        }


Confocal = ConfocalModality
