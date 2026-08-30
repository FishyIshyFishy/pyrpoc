from __future__ import annotations

import time

import numpy as np

from pyrpoc.backend_utils.acquired_data import AcquiredData, DataKind
from pyrpoc.instruments.time_tagger import TimeTaggerInstrument
from pyrpoc.optocontrols.mask import MaskOptoControl  # used by allowed_optocontrols

from .acquisition_core import (
    flim_scan,
    flim_intensity,
    read_flim_frame,
)
from ..base_modality import BaseModality
from ..mod_registry import modality_registry
from . import storage
from .parameters import parameter_groups, FlimParameters

# Settle time after the galvo scan finishes so the last photons reach the
# Flim measurement before the frame is read.
frame_settle_s = 5e-3


@modality_registry.register("flim")
class FlimModality(BaseModality):
    modality_key = "flim"
    display_name = "FLIM"
    parameter_groups = parameter_groups
    required_instruments = [TimeTaggerInstrument]
    optional_instruments = []
    allowed_optocontrols = [MaskOptoControl]
    emitted_kinds = [DataKind.INTENSITY_FRAME, DataKind.FLIM_RAW_FRAME]
    allowed_displays = ["tiled_2d", "multichan_overlay"]

    def __init__(self):
        super().__init__()
        self.parameters: FlimParameters  # narrows base type for type checker
        self._frame_idx = 0
        self._tagger_instrument: TimeTaggerInstrument = None
        self._flim = None
        self._pending_flim_frame: np.ndarray | None = None
        self._raw_frames: list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # Configure sub-steps                                                 #
    # ------------------------------------------------------------------ #

    def load_params(self, params: dict) -> None:
        self.parameters = FlimParameters.from_dict(params)
        self._frame_idx = 0

    def load_instruments(self, instruments: dict) -> None:
        tagger = instruments.get(TimeTaggerInstrument)
        if tagger is None:
            raise RuntimeError("TimeTaggerInstrument missing during configure")
        self._tagger_instrument = tagger

    # ------------------------------------------------------------------ #
    # Acquisition lifecycle                                               #
    # ------------------------------------------------------------------ #

    def laser_period_ps(self) -> int:
        return int(round(1e6 / self.parameters.laser_frequency_mhz))

    def setup_tagger(self) -> None:
        """Create the TimeTagger and start the hardware Flim measurement."""
        p = self.parameters
        total_x = p.x_pixels + p.extra_left + p.extra_right
        self._tagger_instrument.create_tagger()
        self._tagger_instrument.configure_for_flim(
            p.laser_channel,
            p.detector_channel,
            p.pixel_channel,
            p.frame_channel,
            p.laser_trigger_v,
            p.detector_trigger_v,
            p.pixel_trigger_v,
            p.frame_trigger_v,
            laser_input_delay_ps=p.laser_input_delay_ps,
        )
        self._flim = self._tagger_instrument.create_flim_measurement(
            p.laser_channel,
            p.detector_channel,
            p.pixel_channel,
            p.frame_channel,
            n_pixels=total_x * p.y_pixels,
            n_bins=p.histogram_bins,
            binwidth_ps=p.histogram_binwidth_ps,
        )

    def teardown_tagger(self) -> None:
        """Stop the Flim measurement and free the TimeTagger."""
        if self._flim is not None:
            try:
                self._flim.stop()
            except Exception:
                pass
            self._flim = None
        if self._tagger_instrument is not None:
            self._tagger_instrument.free_tagger()

    def acquire_once(self, on_data) -> None:
        p = self.parameters
        self.setup_tagger()
        try:
            flim_scan(
                device_name=p.device_name,
                sample_rate_hz=p.sample_rate_hz,
                fast_axis_ao=p.fast_axis_ao,
                slow_axis_ao=p.slow_axis_ao,
                x_pixels=p.x_pixels,
                y_pixels=p.y_pixels,
                extra_left=p.extra_left,
                extra_right=p.extra_right,
                dwell_time_us=p.dwell_time_us,
                fast_axis_offset=p.fast_axis_offset,
                fast_axis_amplitude=p.fast_axis_amplitude,
                slow_axis_offset=p.slow_axis_offset,
                slow_axis_amplitude=p.slow_axis_amplitude,
                frame_trigger_pfi=p.frame_trigger_pfi_line,
                pixel_clock_ctr=p.pixel_clock_ctr,
                pixel_clock_pfi=p.pixel_clock_pfi_line,
            )
            time.sleep(frame_settle_s)
            total_x = p.x_pixels + p.extra_left + p.extra_right
            hist_frame = read_flim_frame(
                self._flim,
                n_bins=p.histogram_bins,
                y_pixels=p.y_pixels,
                total_x_pixels=total_x,
                extra_left=p.extra_left,
                x_pixels=p.x_pixels,
            )

            self._frame_idx += 1
            self.emit_flim_frame(on_data, hist_frame)
        finally:
            self.teardown_tagger()

    def emit_flim_frame(self, on_data, hist_frame: np.ndarray) -> None:
        self._pending_flim_frame = hist_frame
        intensity = flim_intensity(hist_frame)
        on_data(AcquiredData(
            data=intensity[np.newaxis].astype(np.float32),
            kind=DataKind.INTENSITY_FRAME,
            channel_labels=["intensity"],
        ))
        on_data(AcquiredData(
            data=hist_frame,
            kind=DataKind.FLIM_RAW_FRAME,
            channel_labels=["histogram"],
            metadata={
                "laser_period_ps": self.laser_period_ps(),
                "binwidth_ps": self.parameters.histogram_binwidth_ps,
                "n_bins": self.parameters.histogram_bins,
            },
        ))

    def stop(self) -> None:
        """Signal continuous acquisition to stop and clean up the TimeTagger."""
        self._running = False
        self.teardown_tagger()

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
