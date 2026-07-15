from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

import numpy as np

from pyrpoc.acquisition.hardware.daq import DaqUnavailableError, generate_raster_waveform
from pyrpoc.acquisition.hardware.daq_tasks import run_daq, reshape_to_frame, reshape_to_split_frame
from pyrpoc.acquisition.hardware.flim_daq import flim_scan, flim_intensity, read_flim_frame
from pyrpoc.acquisition.hardware.masks import generate_mask_ttl_signals
from pyrpoc.acquisition.hardware.toy_data import (
    generate_toy_confocal_frame,
    generate_toy_split_confocal_frame,
    generate_toy_flim_frame,
)
from pyrpoc.structs.acquired_data import AcquiredData, DataKind
from pyrpoc.structs.commands import Command, FlimScanCommand, RasterScanCommand, SplitScanCommand

# Settle time after a FLIM galvo scan so the last photons reach the Flim
# measurement before the frame is read.
frame_settle_s = 5e-3


@runtime_checkable
class CommandHandler(Protocol):
    """Runs one command against real hardware, returning its results."""

    def run(self, command: Command) -> list[AcquiredData]:
        ...


class HandlerRegistry:
    """Maps a Command subtype to the handler that runs it.

    Keeps the executor backend-agnostic: it looks up a handler by command type
    and calls it, never knowing about the DAQ or the TimeTagger.
    """

    def __init__(self) -> None:
        self.handlers: dict[type[Command], Any] = {}

    def register(self, command_type: type[Command], handler: Any) -> None:
        self.handlers[command_type] = handler

    def handler_for(self, command_type: type[Command]) -> CommandHandler:
        handler = self.handlers.get(command_type)
        if handler is None:
            raise KeyError(f"no handler registered for command type {command_type.__name__}")
        return handler


command_handler_registry = HandlerRegistry()


class RasterScanHandler:
    """Runs one confocal raster: galvo scan + analog read, mask TTL applied."""

    def run(self, command: RasterScanCommand) -> list[AcquiredData]:
        p = command
        pixel_samples = max(1, int(p.dwell_time_us * 1e-6 * p.sample_rate_hz))
        total_x = p.x_pixels + p.extra_left + p.extra_right
        warning: str | None = None
        try:
            waveform = generate_raster_waveform(
                x_pixels=p.x_pixels,
                extra_left=p.extra_left,
                extra_right=p.extra_right,
                y_pixels=p.y_pixels,
                pixel_samples=pixel_samples,
                fast_axis_offset=p.fast_axis_offset,
                fast_axis_amplitude=p.fast_axis_amplitude,
                slow_axis_offset=p.slow_axis_offset,
                slow_axis_amplitude=p.slow_axis_amplitude,
            )
            ttl_signals = generate_mask_ttl_signals(
                total_x=total_x,
                total_y=p.y_pixels,
                pixel_samples=pixel_samples,
                extra_left=p.extra_left,
                extra_right=p.extra_right,
                device_name=p.device_name,
                mask_contexts=p.mask_contexts,
                scan_x_pixels=p.x_pixels,
            )
            scan_data, total_y_out, x_out, px_out = run_daq(
                device_name=p.device_name,
                sample_rate_hz=p.sample_rate_hz,
                fast_axis_ao=p.fast_axis_ao,
                slow_axis_ao=p.slow_axis_ao,
                waveform=waveform,
                ttl_signals=ttl_signals,
                x_pixels=p.x_pixels,
                y_pixels=p.y_pixels,
                extra_left=p.extra_left,
                extra_right=p.extra_right,
                dwell_time_us=p.dwell_time_us,
                active_ai_channels=list(p.active_ai_channels),
            )
            frame = reshape_to_frame(scan_data, total_y_out, x_out, px_out)
        except DaqUnavailableError:
            warning = "DAQ unavailable — displaying simulated data"
            frame = generate_toy_confocal_frame(
                x_pixels=p.x_pixels,
                y_pixels=p.y_pixels,
                active_channels=list(p.active_ai_channels),
                frame_index=p.frame_index,
                mask_contexts=p.mask_contexts,
                fast_axis_offset=p.fast_axis_offset,
                fast_axis_amplitude=p.fast_axis_amplitude,
                slow_axis_offset=p.slow_axis_offset,
                slow_axis_amplitude=p.slow_axis_amplitude,
            )
        metadata = {"warning": warning} if warning else {}
        return [
            AcquiredData(
                data=frame.astype(np.float32, copy=False),
                kind=DataKind.INTENSITY_FRAME,
                channel_labels=list(p.channel_labels),
                metadata=metadata,
            )
        ]


class SplitScanHandler:
    """Runs one split-confocal raster, time-gating each pixel into t0/t2."""

    def run(self, command: SplitScanCommand) -> list[AcquiredData]:
        p = command
        pixel_samples = max(1, int(p.dwell_time_us * 1e-6 * p.sample_rate_hz))
        total_x = p.x_pixels + p.extra_left + p.extra_right
        warning: str | None = None
        try:
            waveform = generate_raster_waveform(
                x_pixels=p.x_pixels,
                extra_left=p.extra_left,
                extra_right=p.extra_right,
                y_pixels=p.y_pixels,
                pixel_samples=pixel_samples,
                fast_axis_offset=p.fast_axis_offset,
                fast_axis_amplitude=p.fast_axis_amplitude,
                slow_axis_offset=p.slow_axis_offset,
                slow_axis_amplitude=p.slow_axis_amplitude,
            )
            ttl_signals = generate_mask_ttl_signals(
                total_x=total_x,
                total_y=p.y_pixels,
                pixel_samples=pixel_samples,
                extra_left=p.extra_left,
                extra_right=p.extra_right,
                device_name=p.device_name,
                mask_contexts=p.mask_contexts,
                scan_x_pixels=p.x_pixels,
                t0_samples=p.t0_samples,
            )
            scan_data, total_y_out, x_out, px_out = run_daq(
                device_name=p.device_name,
                sample_rate_hz=p.sample_rate_hz,
                fast_axis_ao=p.fast_axis_ao,
                slow_axis_ao=p.slow_axis_ao,
                waveform=waveform,
                ttl_signals=ttl_signals,
                x_pixels=p.x_pixels,
                y_pixels=p.y_pixels,
                extra_left=p.extra_left,
                extra_right=p.extra_right,
                dwell_time_us=p.dwell_time_us,
                active_ai_channels=list(p.active_ai_channels),
            )
            frame, raw = reshape_to_split_frame(
                scan_data, total_y_out, x_out, px_out, p.t0_samples, p.t1_samples
            )
        except (DaqUnavailableError, RuntimeError) as exc:
            warning = f"DAQ unavailable — displaying simulated data ({exc})"
            frame, raw = generate_toy_split_confocal_frame(
                x_pixels=p.x_pixels,
                y_pixels=p.y_pixels,
                active_channels=list(p.active_ai_channels),
                frame_index=p.frame_index,
                mask_contexts=p.mask_contexts,
                fast_axis_offset=p.fast_axis_offset,
                fast_axis_amplitude=p.fast_axis_amplitude,
                slow_axis_offset=p.slow_axis_offset,
                slow_axis_amplitude=p.slow_axis_amplitude,
                t0_samples=p.t0_samples,
                t1_samples=p.t1_samples,
                pixel_samples=pixel_samples,
            )
        metadata: dict = {"auxiliary": {"raw_pixel_stream": raw}}
        if warning:
            metadata["warning"] = warning
        return [
            AcquiredData(
                data=frame.astype(np.float32, copy=False),
                kind=DataKind.INTENSITY_FRAME,
                channel_labels=list(p.channel_labels),
                metadata=metadata,
            )
        ]


class FlimScanHandler:
    """Runs one FLIM raster and reads its per-pixel decay histograms."""

    def run(self, command: FlimScanCommand) -> list[AcquiredData]:
        p = command
        warning: str | None = None
        try:
            if p.simulated or p.flim is None:
                raise DaqUnavailableError("TimeTagger unavailable")
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
                p.flim,
                n_bins=p.histogram_bins,
                y_pixels=p.y_pixels,
                total_x_pixels=total_x,
                extra_left=p.extra_left,
                x_pixels=p.x_pixels,
            )
        except DaqUnavailableError:
            warning = "DAQ unavailable — displaying simulated FLIM data"
            hist_frame = generate_toy_flim_frame(
                x_pixels=p.x_pixels,
                y_pixels=p.y_pixels,
                n_bins=p.histogram_bins,
                binwidth_ps=p.histogram_binwidth_ps,
                laser_period_ps=p.laser_period_ps,
                frame_index=p.frame_index,
                fast_axis_offset=p.fast_axis_offset,
                fast_axis_amplitude=p.fast_axis_amplitude,
                slow_axis_offset=p.slow_axis_offset,
                slow_axis_amplitude=p.slow_axis_amplitude,
            )
        intensity = flim_intensity(hist_frame)
        intensity_meta = {"warning": warning} if warning else {}
        return [
            AcquiredData(
                data=intensity[np.newaxis].astype(np.float32),
                kind=DataKind.INTENSITY_FRAME,
                channel_labels=["intensity"],
                metadata=intensity_meta,
            ),
            AcquiredData(
                data=hist_frame,
                kind=DataKind.FLIM_RAW_FRAME,
                channel_labels=["histogram"],
                metadata={
                    "laser_period_ps": p.laser_period_ps,
                    "binwidth_ps": p.histogram_binwidth_ps,
                    "n_bins": p.histogram_bins,
                },
            ),
        ]


command_handler_registry.register(RasterScanCommand, RasterScanHandler())
command_handler_registry.register(SplitScanCommand, SplitScanHandler())
command_handler_registry.register(FlimScanCommand, FlimScanHandler())
