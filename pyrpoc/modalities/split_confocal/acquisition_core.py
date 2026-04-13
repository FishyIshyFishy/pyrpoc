from __future__ import annotations

import os

import numpy as np
import nidaqmx as nx
from nidaqmx.constants import AcquisitionType

from pyrpoc.backend_utils.opto_control_contexts import MaskContext
from pyrpoc.instruments.confocal_daq import ConfocalDAQInstrument
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from pyrpoc.optocontrols.mask import MaskOptoControl

from ..helpers.daq import generate_raster_waveform

class DaqUnavailableError(RuntimeError):
    """Raised when a DAQ-backed acquisition cannot run on this machine."""


# ---------------------------------------------------------------------------
# Optocontrol helpers
# ---------------------------------------------------------------------------

def extract_mask_contexts(opto_controls: list[BaseOptoControl]) -> list[MaskContext]:
    contexts: list[MaskContext] = []
    for control in opto_controls:
        if not isinstance(control, MaskOptoControl):
            continue
        context = control.context
        if not isinstance(context, MaskContext):
            raise TypeError(
                f"{type(control).__name__} must prepare a MaskContext before acquisition"
            )
        contexts.append(context)
    return contexts

# ---------------------------------------------------------------------------
# Mask TTL signal generation
# ---------------------------------------------------------------------------

def _resize_mask_nearest(mask_bool: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    source_h, source_w = mask_bool.shape
    if source_h <= 0 or source_w <= 0:
        return np.zeros((target_h, target_w), dtype=bool)
    y_idx = np.minimum((np.arange(target_h, dtype=np.int64) * source_h) // target_h, source_h - 1)
    x_idx = np.minimum((np.arange(target_w, dtype=np.int64) * source_w) // target_w, source_w - 1)
    return mask_bool[np.ix_(y_idx, x_idx)]


def _preprocess_mask_to_scan_grid(
    raw_mask: object,
    total_x: int,
    total_y: int,
    scan_x_pixels: int,
    extra_left: int,
    extra_right: int,
) -> np.ndarray:
    if total_x <= 0 or total_y <= 0:
        raise ValueError("total_x and total_y must be positive")
    if (extra_left + scan_x_pixels + extra_right) != total_x:
        raise ValueError("total_x must equal extra_left + scan_x_pixels + extra_right")

    mask = np.asarray(raw_mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape={mask.shape}")

    if scan_x_pixels == 0:
        return np.zeros((total_y, total_x), dtype=bool)

    mask_bool = mask > 0
    if mask_bool.shape != (total_y, scan_x_pixels):
        mask_bool = _resize_mask_nearest(mask_bool, target_h=total_y, target_w=scan_x_pixels)

    padded = np.zeros((total_y, total_x), dtype=bool)
    padded[:, extra_left : extra_left + scan_x_pixels] = mask_bool
    return padded


def _generate_mask_ttl_signals(
    total_x: int,
    total_y: int,
    pixel_samples: int,
    extra_left: int,
    extra_right: int,
    device_name: str,
    mask_contexts: list[MaskContext],
    scan_x_pixels: int,
    t0_samples: int,
) -> dict[str, np.ndarray]:
    """Generate mask TTL signals truncated to the first t0_samples of each pixel."""
    ttl_signals: dict[str, np.ndarray] = {}
    for context in mask_contexts:
        if context.mask is None:
            continue
        channel_name = f"{device_name}/port{int(context.daq_port)}/line{int(context.daq_line)}"
        try:
            padded = _preprocess_mask_to_scan_grid(
                context.mask,
                total_x=total_x,
                total_y=total_y,
                scan_x_pixels=scan_x_pixels,
                extra_left=extra_left,
                extra_right=extra_right,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to preprocess mask for {channel_name}: {exc}") from exc
        if not np.any(padded):
            continue
        ttl = np.zeros((total_y, total_x, pixel_samples), dtype=bool)
        ttl[padded] = True
        if t0_samples < pixel_samples:
            ttl[:, :, t0_samples:] = False
        ttl_signals[channel_name] = ttl.reshape(-1)
    return ttl_signals


# ---------------------------------------------------------------------------
# Raw DAQ acquire + reshape
# ---------------------------------------------------------------------------

def _extract_kept_samples(
    channel_data: np.ndarray,
    total_y: int,
    total_x: int,
    pixel_samples: int,
    extra_left: int,
    x_pixels: int,
) -> np.ndarray:
    scan_line = np.asarray(channel_data, dtype=np.float32).reshape(total_y, total_x * pixel_samples)
    pixel_grid = scan_line.reshape(total_y, total_x, pixel_samples)
    kept = pixel_grid[:, extra_left : extra_left + x_pixels, :]
    return kept.reshape(total_y, x_pixels * pixel_samples).astype(np.float32, copy=False)


def _run_daq(
    daq_instrument: ConfocalDAQInstrument,
    waveform: np.ndarray,
    ttl_signals: dict[str, np.ndarray],
    x_pixels: int,
    y_pixels: int,
    extra_left: int,
    extra_right: int,
    dwell_time_us: float,
    active_ai_channels: list[int],
) -> tuple[np.ndarray, int, int, int]:
    device_name = daq_instrument.device_name
    sample_rate_hz = float(daq_instrument.sample_rate_hz)
    fast_axis_channel = int(daq_instrument.fast_axis_ao)
    slow_axis_channel = int(daq_instrument.slow_axis_ao)

    pixel_samples = max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))
    total_x = x_pixels + extra_left + extra_right
    total_y = y_pixels
    total_samples = total_x * total_y * pixel_samples

    ai_channels = [f"{device_name}/ai{idx}" for idx in active_ai_channels]
    do_task: nx.Task | None = None
    static_do_task: nx.Task | None = None
    static_values: list[bool] = []

    try:
        with nx.Task() as ao_task, nx.Task() as ai_task:
            ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_axis_channel}")
            ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_axis_channel}")
            for ch in ai_channels:
                ai_task.ai_channels.add_ai_voltage_chan(ch)

            ao_task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=total_samples,
            )
            ai_task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                source=f"/{device_name}/ao/SampleClock",
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=total_samples,
            )

            if ttl_signals:
                dynamic_channels, dynamic_ttls = [], []
                static_channels = []

                for channel_name, ttl in ttl_signals.items():
                    if "/port0/" in channel_name.lower():
                        dynamic_channels.append(channel_name)
                        dynamic_ttls.append(ttl)
                    else:
                        static_channels.append(channel_name)
                        static_values.append(bool(ttl.flat[0]))

                if dynamic_channels:
                    do_task = nx.Task()
                    for ch in dynamic_channels:
                        do_task.do_channels.add_do_chan(ch)
                    do_task.timing.cfg_samp_clk_timing(
                        rate=sample_rate_hz,
                        source=f"/{device_name}/ao/SampleClock",
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=total_samples,
                    )
                    payload = dynamic_ttls[0].tolist() if len(dynamic_channels) == 1 else [t.tolist() for t in dynamic_ttls]
                    do_task.write(payload, auto_start=False)  # pyright:ignore

                if static_channels:
                    static_do_task = nx.Task()
                    for ch in static_channels:
                        static_do_task.do_channels.add_do_chan(ch)
                    static_do_task.write(static_values, auto_start=True)  # pyright:ignore

            ao_task.write(np.asarray(waveform, dtype=np.float64), auto_start=False)  # pyright:ignore
            ai_task.start()
            if do_task is not None:
                do_task.start()
            ao_task.start()

            timeout = total_samples / sample_rate_hz + 5
            ao_task.wait_until_done(timeout=timeout)
            ai_task.wait_until_done(timeout=timeout)
            if do_task is not None:
                do_task.wait_until_done(timeout=timeout)

            acq_data = np.asarray(ai_task.read(number_of_samples_per_channel=total_samples), dtype=np.float32)  # pyright:ignore
            if acq_data.ndim == 1:
                acq_data = acq_data[np.newaxis, :]
            elif acq_data.ndim != 2:
                raise RuntimeError("Unexpected NI-DAQ data shape")

            channels_out = [
                _extract_kept_samples(ch_data, total_y, total_x, pixel_samples, extra_left, x_pixels)
                for ch_data in acq_data
            ]
            return np.stack(channels_out, axis=0).astype(np.float32, copy=False), total_y, x_pixels, pixel_samples

    except Exception as exc:
        raise DaqUnavailableError(f"NI-DAQ acquisition failed: {exc}") from exc
    finally:
        if do_task is not None:
            do_task.close()
        if static_do_task is not None:
            if static_values:
                try:
                    static_do_task.write([not v for v in static_values], auto_start=True)  # pyright:ignore
                except Exception:
                    pass
            static_do_task.close()


def _reshape_to_split_frame(
    scan_data: np.ndarray,
    total_y: int,
    x_pixels: int,
    pixel_samples: int,
    t0_samples: int,
    t1_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    split_point = int(t0_samples)
    second_start = split_point + int(t1_samples)

    split_channels: list[np.ndarray] = []
    raw_channels: list[np.ndarray] = []

    for ch_data in scan_data:
        pixel_data = np.asarray(ch_data, dtype=np.float32).reshape(total_y, x_pixels, pixel_samples)
        raw_channels.append(pixel_data.astype(np.float32, copy=False))

        first_half = pixel_data[:, :, :split_point].mean(axis=2)
        second_half = (
            pixel_data[:, :, second_start:].mean(axis=2)
            if second_start < pixel_samples
            else np.zeros_like(first_half)
        )
        split_channels.append(first_half.astype(np.float32, copy=False))
        split_channels.append(second_half.astype(np.float32, copy=False))

    return (
        np.stack(split_channels, axis=0).astype(np.float32, copy=False),
        np.stack(raw_channels, axis=0).astype(np.float32, copy=False),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def acquire_frame(
    daq_instrument: ConfocalDAQInstrument,
    x_pixels: int,
    y_pixels: int,
    extra_left: int,
    extra_right: int,
    dwell_time_us: float,
    fast_axis_offset: float,
    fast_axis_amplitude: float,
    slow_axis_offset: float,
    slow_axis_amplitude: float,
    active_ai_channels: list[int],
    t0_samples: int,
    t1_samples: int,
    mask_contexts: list[MaskContext],
) -> tuple[np.ndarray, np.ndarray]:
    """Perform one split-confocal raster scan.

    Returns ``(split_frame, raw_frame)`` where ``split_frame`` is (C*2, H, W)
    with alternating t0/t2 channels, and ``raw_frame`` is (C, H, W) raw pixel
    samples.
    """
    sample_rate_hz = float(daq_instrument.sample_rate_hz)
    pixel_samples = max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))
    total_x = x_pixels + extra_left + extra_right

    waveform = generate_raster_waveform(
        x_pixels=x_pixels,
        extra_left=extra_left,
        extra_right=extra_right,
        y_pixels=y_pixels,
        pixel_samples=pixel_samples,
        fast_axis_offset=fast_axis_offset,
        fast_axis_amplitude=fast_axis_amplitude,
        slow_axis_offset=slow_axis_offset,
        slow_axis_amplitude=slow_axis_amplitude,
    )
    ttl_signals = _generate_mask_ttl_signals(
        total_x=total_x,
        total_y=y_pixels,
        pixel_samples=pixel_samples,
        extra_left=extra_left,
        extra_right=extra_right,
        device_name=daq_instrument.device_name,
        mask_contexts=mask_contexts,
        scan_x_pixels=x_pixels,
        t0_samples=t0_samples,
    )
    scan_data, total_y_out, x_out, px_out = _run_daq(
        daq_instrument=daq_instrument,
        waveform=waveform,
        ttl_signals=ttl_signals,
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        extra_left=extra_left,
        extra_right=extra_right,
        dwell_time_us=dwell_time_us,
        active_ai_channels=active_ai_channels,
    )
    return _reshape_to_split_frame(scan_data, total_y_out, x_out, px_out, t0_samples, t1_samples)
