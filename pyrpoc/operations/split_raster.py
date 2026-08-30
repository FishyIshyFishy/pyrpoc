"""Split confocal: the same scan, with each pixel's samples split into windows.

Moved from ``modalities/split_confocal/acquisition_core.py``. Everything that
was duplicated verbatim from the confocal copy — ``run_daq``,
``extract_kept_samples``, ``resize_mask_nearest``,
``preprocess_mask_to_scan_grid`` — is not here; it is in ``raster.py`` and
``modulation.py`` and this module calls it. What is here is what genuinely
differs: splitting a pixel's samples into a t0 window and a t2 window, and
returning the raw sample stack alongside.
"""

from __future__ import annotations

import numpy as np

from .raster import generate_raster_waveform, pixel_samples, run_raster


def reshape_to_split_frame(
    scan_data: np.ndarray,
    total_y: int,
    x_pixels: int,
    pixel_samples: int,
    t0_samples: int,
    t1_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split each pixel's samples into ``t0`` and ``t2`` means.

    Returns ``(split, raw)`` where ``split`` is ``(C*2, H, W)`` with alternating
    t0/t2 channels and ``raw`` is ``(C, H, W, S)`` of unaveraged samples.
    """
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


def split_raster_scan(
    *,
    x_pixels: int,
    y_pixels: int,
    extra_left: int,
    extra_right: int,
    fast_axis_offset: float,
    fast_axis_amplitude: float,
    slow_axis_offset: float,
    slow_axis_amplitude: float,
    dwell_time_us: float,
    sample_rate_hz: float,
    device_name: str,
    ai_channels: tuple[int, ...] | list[int],
    fast_ao: int,
    slow_ao: int,
    t0_samples: int,
    t1_samples: int,
    ttl: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One split-confocal raster scan. Returns ``(split_frame, raw_frame)``."""
    samples_per_pixel = pixel_samples(dwell_time_us, sample_rate_hz)

    waveform = generate_raster_waveform(
        x_pixels=x_pixels,
        extra_left=extra_left,
        extra_right=extra_right,
        y_pixels=y_pixels,
        pixel_samples=samples_per_pixel,
        fast_axis_offset=fast_axis_offset,
        fast_axis_amplitude=fast_axis_amplitude,
        slow_axis_offset=slow_axis_offset,
        slow_axis_amplitude=slow_axis_amplitude,
    )
    scan_data, total_y_out, x_out, px_out = run_raster(
        device_name=device_name,
        sample_rate_hz=sample_rate_hz,
        fast_ao=fast_ao,
        slow_ao=slow_ao,
        waveform=waveform,
        ttl_signals=ttl or {},
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        extra_left=extra_left,
        extra_right=extra_right,
        dwell_time_us=dwell_time_us,
        ai_channels=list(ai_channels),
    )
    return reshape_to_split_frame(scan_data, total_y_out, x_out, px_out, t0_samples, t1_samples)
