from __future__ import annotations

from typing import Any

import numpy as np

from pyrpoc.backend_utils.array_contracts import CONTRACT_CHW_FLOAT32
from pyrpoc.instruments.confocal_daq import ConfocalDAQInstrument
from pyrpoc.backend_utils.opto_control_contexts import MaskContext
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from pyrpoc.optocontrols.mask import MaskOptoControl

from ..acquisition_functions.daq_helpers import (
    DaqUnavailableError,
    acquire_daq_split_confocal,
    extract_mask_contexts,
)
from ..acquisition_functions.toy_data import generate_toy_split_confocal_frame
from ..base_modality import BaseModality
from ..mod_registry import modality_registry
from . import storage
from .parameters import PARAMETERS, SplitConfocalParameters


@modality_registry.register("split_confocal")
class SplitConfocalModality(BaseModality):
    MODALITY_KEY = "split_confocal"
    DISPLAY_NAME = "Split Confocal"
    PARAMETERS = PARAMETERS
    REQUIRED_INSTRUMENTS = [ConfocalDAQInstrument]
    OPTIONAL_INSTRUMENTS = []
    ALLOWED_OPTOCONTROLS = [MaskOptoControl]
    OUTPUT_DATA_CONTRACT = CONTRACT_CHW_FLOAT32
    ALLOWED_DISPLAYS = ["tiled_2d", "multichan_overlay"]

    def __init__(self):
        super().__init__()
        self.parameters: SplitConfocalParameters  # narrows base type for type checker
        self._frame_idx = 0
        self._mask_contexts: list[MaskContext] = []
        self._daq_instrument: ConfocalDAQInstrument = None
        self._active_ai_channels: list[int] = []
        self._pending_auxiliary: dict[str, np.ndarray] = {}
        self._auxiliary_payload_buffers: dict[str, list[np.ndarray]] = {}
        self._auxiliary_paths: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Configure sub-steps                                                 #
    # ------------------------------------------------------------------ #

    def load_params(self, params: dict) -> None:
        self.parameters = SplitConfocalParameters.from_dict(params)
        self._frame_idx = 0
        self._pending_auxiliary = {}

    def load_instruments(self, instruments: dict) -> None:
        instrument = instruments.get(ConfocalDAQInstrument)
        if instrument is None:
            raise RuntimeError("ConfocalDAQInstrument missing during configure")
        self._daq_instrument = instrument
        self._active_ai_channels = self._daq_instrument.report_active_ai_channels()

    def load_optocontrols(self, opto_controls: list[BaseOptoControl]) -> None:
        self._mask_contexts = extract_mask_contexts(opto_controls)

    # ------------------------------------------------------------------ #
    # Acquisition lifecycle                                               #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if not self._configured:
            raise RuntimeError("modality must be configured before start")
        if self._daq_instrument is None:
            raise RuntimeError("ConfocalDAQInstrument not configured")
        self._running = True

    def acquire_once(self) -> np.ndarray:
        if not self._running:
            raise RuntimeError("modality is not running")

        p = self.parameters
        try:
            frame, raw = acquire_daq_split_confocal(
                daq_instrument=self._daq_instrument,
                active_ai_channels=self._active_ai_channels,
                mask_contexts=self._mask_contexts,
                x_pixels=p.x_pixels,
                y_pixels=p.y_pixels,
                extra_left=p.extra_left,
                extra_right=p.extra_right,
                dwell_time_us=p.dwell_time_us,
                fast_axis_offset=p.fast_axis_offset,
                fast_axis_amplitude=p.fast_axis_amplitude,
                slow_axis_offset=p.slow_axis_offset,
                slow_axis_amplitude=p.slow_axis_amplitude,
                t0_samples=p.t0_samples,
                t1_samples=p.t1_samples,
            )
        except Exception as exc:
            if not isinstance(exc, (DaqUnavailableError, RuntimeError)):
                raise
            self._emit_warning(f"DAQ unavailable — displaying simulated data ({exc})")
            pixel_samples = max(1, int(p.dwell_time_us * 1e-6 * float(self._daq_instrument.sample_rate_hz)))
            frame, raw = generate_toy_split_confocal_frame(
                x_pixels=p.x_pixels,
                y_pixels=p.y_pixels,
                active_channels=self._active_ai_channels,
                frame_index=self._frame_idx,
                mask_contexts=self._mask_contexts,
                fast_axis_offset=p.fast_axis_offset,
                fast_axis_amplitude=p.fast_axis_amplitude,
                slow_axis_offset=p.slow_axis_offset,
                slow_axis_amplitude=p.slow_axis_amplitude,
                t0_samples=p.t0_samples,
                t1_samples=p.t1_samples,
                pixel_samples=pixel_samples,
            )

        self._pending_auxiliary = {"raw_pixel_stream": raw}
        self._frame_idx += 1
        return frame.astype(np.float32, copy=False)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------ #
    # Storage delegation                                                  #
    # ------------------------------------------------------------------ #

    def prepare_acquisition_storage(self, *, frame_limit: int | None) -> None:
        storage.prepare_acquisition_storage(self, frame_limit=frame_limit)

    def save_acquired_frame(self, frame: np.ndarray, *, frame_index: int) -> None:
        storage.save_acquired_frame(self, frame, frame_index=frame_index)

    def finalize_acquisition_storage(self, *, frame_count: int, frame_limit: int | None, error: Exception | None) -> None:
        storage.finalize_acquisition_storage(self, frame_count=frame_count, frame_limit=frame_limit, error=error)

    # ------------------------------------------------------------------ #
    # Channel labels                                                      #
    # ------------------------------------------------------------------ #

    def get_active_channel_labels(self) -> list[str]:
        labels: list[str] = []
        for ch in self._active_ai_channels:
            labels.append(f"ai{ch}_t0")
            labels.append(f"ai{ch}_t2")
        return labels

SplitConfocal = SplitConfocalModality
