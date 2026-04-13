from __future__ import annotations

import threading
from typing import Any

import numpy as np

from pyrpoc.backend_utils.acquired_data import AcquiredData, DataKind
from pyrpoc.instruments.confocal_daq import ConfocalDAQInstrument
from pyrpoc.instruments.time_tagger import TimeTaggerInstrument
from pyrpoc.backend_utils.opto_control_contexts import MaskContext
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from pyrpoc.optocontrols.mask import MaskOptoControl  # used by load_optocontrols type signature

from .acquisition_core import (
    DaqUnavailableError,
    acquire_daq_frame,
    poll_one_flim_frame,
)
from ..helpers.toy_data import generate_toy_confocal_frame
from ..base_modality import BaseModality
from ..mod_registry import modality_registry
from . import storage
from .parameters import PARAMETERS, FlimParameters


@modality_registry.register("flim")
class FlimModality(BaseModality):
    MODALITY_KEY = "flim"
    DISPLAY_NAME = "FLIM"
    PARAMETERS = PARAMETERS
    REQUIRED_INSTRUMENTS = [ConfocalDAQInstrument, TimeTaggerInstrument]
    OPTIONAL_INSTRUMENTS = []
    ALLOWED_OPTOCONTROLS = [MaskOptoControl]
    EMITTED_KINDS = [DataKind.INTENSITY_FRAME, DataKind.PARTIAL_FRAME]
    ALLOWED_DISPLAYS = ["tiled_2d", "multichan_overlay"]

    def __init__(self):
        super().__init__()
        self.parameters: FlimParameters  # narrows base type for type checker
        self._frame_idx = 0
        self._mask_contexts: list[MaskContext] = []
        self._daq_instrument: ConfocalDAQInstrument = None
        self._tagger_instrument: TimeTaggerInstrument = None
        self._stream = None
        self._active_ai_channels: list[int] = []
        self._pending_flim_frame: np.ndarray | None = None
        self._raw_frames: list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # Configure sub-steps                                                 #
    # ------------------------------------------------------------------ #

    def load_params(self, params: dict) -> None:
        self.parameters = FlimParameters.from_dict(params)
        self._frame_idx = 0

    def load_instruments(self, instruments: dict) -> None:
        daq = instruments.get(ConfocalDAQInstrument)
        tagger = instruments.get(TimeTaggerInstrument)
        if daq is None:
            raise RuntimeError("ConfocalDAQInstrument missing during configure")
        if tagger is None:
            raise RuntimeError("TimeTaggerInstrument missing during configure")
        self._daq_instrument = daq
        self._tagger_instrument = tagger
        self._active_ai_channels = self._daq_instrument.report_active_ai_channels()

    # ------------------------------------------------------------------ #
    # Acquisition lifecycle                                               #
    # ------------------------------------------------------------------ #

    def _setup_tagger(self) -> None:
        """Create and configure the TimeTagger and FLIM stream for one acquisition."""
        p = self.parameters
        self._tagger_instrument.create_tagger()
        self._tagger_instrument.configure_for_flim(
            laser_ch=p.laser_channel,
            detector_ch=p.detector_channel,
            trigger_ch=p.daq_trigger_channel,
            laser_trigger_v=p.laser_trigger_v,
            detector_trigger_v=p.detector_trigger_v,
            trigger_v=p.trigger_v,
            laser_event_divider=p.laser_event_divider,
        )
        self._stream = self._tagger_instrument.create_flim_stream(
            laser_ch=p.laser_channel,
            detector_ch=p.detector_channel,
            trigger_ch=p.daq_trigger_channel,
        )
        self._stream.start()

    def _teardown_tagger(self) -> None:
        """Stop the FLIM stream and free the TimeTagger."""
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            self._stream = None
        if self._tagger_instrument is not None:
            self._tagger_instrument.free_tagger()

    def acquire_once(self, on_data) -> None:
        p = self.parameters
        pixel_dwell_ps = int(round(p.dwell_time_us * 1e6))
        scan_duration_s = ((p.x_pixels+p.extra_left+p.extra_right) * p.y_pixels * p.dwell_time_us * 1e-6)

        def _partial_callback(partial_array: np.ndarray) -> None:
            on_data(AcquiredData(
                data=partial_array,
                kind=DataKind.PARTIAL_FRAME,
                channel_labels=["intensity"],
            ))

        self._setup_tagger()
        try:
            poll_result: dict[str, Any] = {}

            def _poll() -> None:
                try:
                    poll_result["frame"] = poll_one_flim_frame(
                        stream=self._stream,
                        x_pixels=p.x_pixels,
                        y_pixels=p.y_pixels,
                        extra_left=p.extra_left,
                        extra_right=p.extra_right,
                        pixel_dwell_ps=pixel_dwell_ps,
                        laser_ch=p.laser_channel,
                        detector_ch=p.detector_channel,
                        trigger_ch=p.daq_trigger_channel,
                        progress_callback=_partial_callback,
                    )
                except Exception as exc:
                    poll_result["error"] = exc

            poll_thread = threading.Thread(target=_poll, daemon=True)
            poll_thread.start()

            try:
                acquire_daq_frame(
                    daq_instrument=self._daq_instrument,
                    x_pixels=p.x_pixels,
                    y_pixels=p.y_pixels,
                    extra_left=p.extra_left,
                    extra_right=p.extra_right,
                    dwell_time_us=p.dwell_time_us,
                    fast_axis_offset=p.fast_axis_offset,
                    fast_axis_amplitude=p.fast_axis_amplitude,
                    slow_axis_offset=p.slow_axis_offset,
                    slow_axis_amplitude=p.slow_axis_amplitude,
                    active_ai_channels=self._active_ai_channels,
                    daq_trigger_pfi_line=p.daq_trigger_pfi_line,
                )
            except DaqUnavailableError:
                self._emit_warning("DAQ unavailable — displaying simulated data")
                generate_toy_confocal_frame(
                    x_pixels=p.x_pixels,
                    y_pixels=p.y_pixels,
                    active_channels=self._active_ai_channels if self._active_ai_channels else [0],
                    frame_index=self._frame_idx,
                    mask_contexts=self._mask_contexts,
                    fast_axis_offset=p.fast_axis_offset,
                    fast_axis_amplitude=p.fast_axis_amplitude,
                    slow_axis_offset=p.slow_axis_offset,
                    slow_axis_amplitude=p.slow_axis_amplitude,
                )

            self._frame_idx += 1

            poll_thread.join(timeout=scan_duration_s * 2 + 5)
            if "error" in poll_result:
                raise RuntimeError(f"TimeTagger poll failed: {poll_result['error']}") from poll_result["error"]

            flim_frame = poll_result.get("frame")
            self._pending_flim_frame = flim_frame
            if flim_frame is None:
                raise RuntimeError("TimeTagger poll did not return a frame within the expected window")
            intensity = np.vectorize(len)(flim_frame).astype(np.float32)
            on_data(AcquiredData(
                data=intensity[np.newaxis].astype(np.float32),
                kind=DataKind.INTENSITY_FRAME,
                channel_labels=["intensity"],
            ))
        finally:
            self._teardown_tagger()

    def stop(self) -> None:
        """Signal continuous acquisition to stop and clean up any hardware
        that may have been left mid-frame (e.g. stream interrupted by error)."""
        self._running = False
        self._teardown_tagger()

    # ------------------------------------------------------------------ #
    # Storage delegation                                                  #
    # ------------------------------------------------------------------ #

    def prepare_acquisition_storage(self, *, frame_limit: int | None) -> None:
        storage.prepare_acquisition_storage(self, frame_limit=frame_limit)

    def save_acquired_frame(self, acquired: AcquiredData, *, frame_index: int) -> None:
        storage.save_acquired_frame(self, acquired.data, frame_index=frame_index)

    def finalize_acquisition_storage(self, *, frame_count: int, frame_limit: int | None, error: Exception | None) -> None:
        storage.finalize_acquisition_storage(self, frame_count=frame_count, frame_limit=frame_limit, error=error)

    def get_active_channel_labels(self) -> list[str]:
        return ["intensity"]


Flim = FlimModality
