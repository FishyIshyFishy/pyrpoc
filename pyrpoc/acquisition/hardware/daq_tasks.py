from __future__ import annotations

import numpy as np
import nidaqmx as nx
from nidaqmx.constants import AcquisitionType

from pyrpoc.acquisition.hardware.daq import DaqUnavailableError


def extract_kept_samples(
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


def run_daq(
    device_name: str,
    sample_rate_hz: float,
    fast_axis_ao: int,
    slow_axis_ao: int,
    waveform: np.ndarray,
    ttl_signals: dict[str, np.ndarray],
    x_pixels: int,
    y_pixels: int,
    extra_left: int,
    extra_right: int,
    dwell_time_us: float,
    active_ai_channels: list[int],
) -> tuple[np.ndarray, int, int, int]:
    """Run one clocked galvo raster (AO) + analog read (AI) + optional mask DO.

    Returns ``(scan_data, total_y, x_pixels, pixel_samples)`` where scan_data is
    the per-channel kept samples. Raises DaqUnavailableError if the hardware is
    not reachable.
    """
    fast_axis_channel = int(fast_axis_ao)
    slow_axis_channel = int(slow_axis_ao)

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
                extract_kept_samples(ch_data, total_y, total_x, pixel_samples, extra_left, x_pixels)
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


def reshape_to_frame(
    scan_data: np.ndarray,
    total_y: int,
    x_pixels: int,
    pixel_samples: int,
) -> np.ndarray:
    frame_channels = [
        np.asarray(ch, dtype=np.float32).reshape(total_y, x_pixels, pixel_samples).mean(axis=2)
        for ch in scan_data
    ]
    return np.stack(frame_channels, axis=0).astype(np.float32, copy=False)


def reshape_to_split_frame(
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
