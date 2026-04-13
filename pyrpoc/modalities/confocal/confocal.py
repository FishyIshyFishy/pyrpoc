from __future__ import annotations

import numpy as np

from pyrpoc.backend_utils.acquired_data import AcquiredData, DataKind
from pyrpoc.instruments.confocal_daq import ConfocalDAQInstrument
from pyrpoc.backend_utils.opto_control_contexts import MaskContext
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from pyrpoc.optocontrols.mask import MaskOptoControl

from .acquisition_core import (
    DaqUnavailableError,
    acquire_frame as acquire_daq_confocal,
    extract_mask_contexts,
)
from ..helpers.toy_data import generate_toy_confocal_frame
from ..base_modality import BaseModality
from ..mod_registry import modality_registry
from . import storage
from .parameters import PARAMETERS, ConfocalParameters


@modality_registry.register("confocal")
class ConfocalModality(BaseModality):
    MODALITY_KEY = "confocal"
    DISPLAY_NAME = "Confocal"
    PARAMETERS = PARAMETERS
    REQUIRED_INSTRUMENTS = [ConfocalDAQInstrument]
    OPTIONAL_INSTRUMENTS = []
    ALLOWED_OPTOCONTROLS = [MaskOptoControl]
    EMITTED_KINDS = [DataKind.INTENSITY_FRAME]
    ALLOWED_DISPLAYS = ["tiled_2d", "multichan_overlay"]

    def __init__(self):
        super().__init__()
        self.parameters: ConfocalParameters  # narrows base type for type checker
        self._frame_idx = 0
        self._mask_contexts: list[MaskContext] = []
        self._daq_instrument: ConfocalDAQInstrument = None
        self._active_ai_channels: list[int] = []

    # ------------------------------------------------------------------ #
    # Configure sub-steps                                                 #
    # ------------------------------------------------------------------ #

    def load_params(self, params: dict) -> None:
        self.parameters = ConfocalParameters.from_dict(params)
        self._frame_idx = 0

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

    def acquire_once(self, on_data) -> None:
        p = self.parameters
        try:
            frame = acquire_daq_confocal(
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
            )
        except DaqUnavailableError:
            self._emit_warning("DAQ unavailable — displaying simulated data")
            frame = generate_toy_confocal_frame(
                x_pixels=p.x_pixels,
                y_pixels=p.y_pixels,
                active_channels=self._active_ai_channels,
                frame_index=self._frame_idx,
                mask_contexts=self._mask_contexts,
                fast_axis_offset=p.fast_axis_offset,
                fast_axis_amplitude=p.fast_axis_amplitude,
                slow_axis_offset=p.slow_axis_offset,
                slow_axis_amplitude=p.slow_axis_amplitude,
            )
        self._frame_idx += 1
        on_data(AcquiredData(
            data=frame.astype(np.float32, copy=False),
            kind=DataKind.INTENSITY_FRAME,
            channel_labels=self.get_active_channel_labels(),
        ))

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------ #
    # Storage delegation                                                  #
    # ------------------------------------------------------------------ #

    def prepare_acquisition_storage(self, *, frame_limit: int | None) -> None:
        storage.prepare_acquisition_storage(self, frame_limit=frame_limit)

    def save_acquired_frame(self, acquired: AcquiredData, *, frame_index: int) -> None:
        storage.save_acquired_frame(self, acquired.data, frame_index=frame_index)

    def finalize_acquisition_storage(self, *, frame_count: int, frame_limit: int | None, error: Exception | None) -> None:
        storage.finalize_acquisition_storage(self, frame_count=frame_count, frame_limit=frame_limit, error=error)

    # ------------------------------------------------------------------ #
    # Channel labels                                                      #
    # ------------------------------------------------------------------ #

    def get_active_channel_labels(self) -> list[str]:
        return [f"ai{ch}" for ch in self._active_ai_channels] if self._active_ai_channels else []


Confocal = ConfocalModality
