from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import nidaqmx as nx
from nidaqmx.constants import AcquisitionType, Signal

from ..helpers.daq import generate_raster_waveform

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class DaqUnavailableError(RuntimeError):
    """Raised when a DAQ-backed acquisition cannot run on this machine."""


# ---------------------------------------------------------------------------
# DAQ: raw hardware acquire
# ---------------------------------------------------------------------------

def reshape_channel(
    ch: np.ndarray,
    y_pixels: int,
    x_pixels: int,
    extra_left: int,
    extra_right: int,
    pixel_samples: int,
) -> np.ndarray:
    total_x = extra_left + x_pixels + extra_right
    pixel_grid = ch.reshape(y_pixels, total_x, pixel_samples)
    return pixel_grid[:, extra_left : extra_left + x_pixels, :].mean(axis=2).astype(np.float32, copy=False)


def run_daq(
    device_name: str,
    sample_rate_hz: float,
    fast_axis_ao: int,
    slow_axis_ao: int,
    raster_waveform: np.ndarray,
    x_pixels: int,
    y_pixels: int,
    extra_left: int,
    extra_right: int,
    dwell_time_us: float,
    active_ai_channels: list[int],
    daq_trigger_pfi_line: int,
) -> np.ndarray:
    fast_axis_channel = int(fast_axis_ao)
    slow_axis_channel = int(slow_axis_ao)

    pixel_samples = max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))
    total_x = x_pixels + extra_left + extra_right
    total_y = y_pixels
    total_samples = total_x * total_y * pixel_samples

    ai_channels = [f"{device_name}/ai{idx}" for idx in active_ai_channels]
    timeout = total_samples / sample_rate_hz + 5
    try:
        with nx.Task() as ao_task, nx.Task() as ai_task:
            ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_axis_channel}")
            ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_axis_channel}")
            for ch in ai_channels:
                ai_task.ai_channels.add_ai_voltage_chan(ch)

            ao_task.timing.cfg_samp_clk_timing(rate=sample_rate_hz, sample_mode=AcquisitionType.FINITE, samps_per_chan=total_samples)
            ai_task.timing.cfg_samp_clk_timing(rate=sample_rate_hz, source=f"/{device_name}/ao/SampleClock", sample_mode=AcquisitionType.FINITE, samps_per_chan=total_samples)

            # Export the AO start trigger to the PFI line so the TimeTagger sees one frame-start pulse
            ao_task.export_signals.export_signal(Signal.START_TRIGGER, f"/{device_name}/PFI{int(daq_trigger_pfi_line)}")

            ao_task.write(np.asarray(raster_waveform, dtype=np.float64), auto_start=False)  # pyright:ignore
            ai_task.start()
            ao_task.start()
            ao_task.wait_until_done(timeout=timeout)
            ai_task.wait_until_done(timeout=timeout)

            acq_data = np.asarray(ai_task.read(number_of_samples_per_channel=total_samples), dtype=np.float32)  # pyright:ignore
            if acq_data.ndim == 1:
                acq_data = acq_data[np.newaxis, :]
            return acq_data.astype(np.float32, copy=False)

    except Exception as exc:
        raise DaqUnavailableError(f"NI-DAQ acquisition failed: {exc}") from exc


def acquire_daq_frame(
    device_name: str,
    sample_rate_hz: float,
    fast_axis_ao: int,
    slow_axis_ao: int,
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
    daq_trigger_pfi_line: int,
) -> np.ndarray:
    """Drive one FLIM raster scan on the DAQ, exporting a frame-start trigger on a PFI line.

    Returns a (C, H, W) float32 intensity frame (DAQ analog channels only).
    Raises DaqUnavailableError if the hardware is not reachable.
    """
    pixel_samples = max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))

    raster_waveform = generate_raster_waveform(
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
    # TODO: add mask functionality back in later

    scan_data = run_daq(
        device_name=device_name,
        sample_rate_hz=sample_rate_hz,
        fast_axis_ao=fast_axis_ao,
        slow_axis_ao=slow_axis_ao,
        raster_waveform=raster_waveform,
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        extra_left=extra_left,
        extra_right=extra_right,
        dwell_time_us=dwell_time_us,
        active_ai_channels=active_ai_channels,
        daq_trigger_pfi_line=daq_trigger_pfi_line,
    )
    frame_channels = [
        reshape_channel(ch=ch, y_pixels=y_pixels, x_pixels=x_pixels,
                        extra_left=extra_left, extra_right=extra_right, pixel_samples=pixel_samples)
        for ch in scan_data
    ]
    return np.stack(frame_channels, axis=0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# TimeTagger: delay calculation
# ---------------------------------------------------------------------------

def _compute_delays(
    detector_ts: np.ndarray,
    laser_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Match each detector hit to its preceding laser pulse.

    Returns (valid_det_ts, delays_ps), or None if no match is possible.
    """
    if not detector_ts.size or not laser_ts.size:
        return None

    indices = np.searchsorted(laser_ts, detector_ts, side="right") - 1
    valid = indices >= 0
    valid_det_ts = detector_ts[valid]
    delays_ps = valid_det_ts - laser_ts[indices[valid]]
    return valid_det_ts, delays_ps


# ---------------------------------------------------------------------------
# TimeTagger: pixel binning
# ---------------------------------------------------------------------------

def _bin_delays_into_pixels(
    valid_det_ts: np.ndarray,
    delays_ps: np.ndarray,
    frame_start_ps: int,
    pixel_dwell_ps: int,
    total_pixels: int,
    pixel_lists: list[list[int]],
) -> None:
    """Append photon delays into the appropriate pixel bucket (in-place)."""
    pixel_indices = (
        (valid_det_ts - frame_start_ps) // int(pixel_dwell_ps)
    ).astype(np.int64, copy=False)

    in_frame = (pixel_indices >= 0) & (pixel_indices < total_pixels)
    if not np.any(in_frame):
        return

    pix = pixel_indices[in_frame]
    dlys = delays_ps[in_frame]
    order = np.argsort(pix, kind="stable")
    pix = pix[order]
    dlys = dlys[order]

    splits = np.flatnonzero(np.diff(pix)) + 1
    for pix_grp, dly_grp in zip(np.split(pix, splits), np.split(dlys, splits)):
        pixel_lists[int(pix_grp[0])].extend(dly_grp.tolist())


# ---------------------------------------------------------------------------
# TimeTagger: frame finalisation
# ---------------------------------------------------------------------------

def _finalize_frame(
    pixel_lists: list[list[int]],
    y_pixels: int,
    total_x_pixels: int,
    extra_left: int,
    x_pixels: int,
) -> np.ndarray:
    frame = np.empty((y_pixels, total_x_pixels), dtype=object)
    for flat_index, delays in enumerate(pixel_lists):
        iy, ix = divmod(flat_index, total_x_pixels)
        frame[iy, ix] = np.asarray(delays, dtype=np.int64)
    return frame[:, extra_left : extra_left + x_pixels]


# ---------------------------------------------------------------------------
# TimeTagger: public poll_one_flim_frame
# ---------------------------------------------------------------------------

def _partial_intensity(
    pixel_lists: list[list[int]],
    y_pixels: int,
    total_x: int,
    extra_left: int,
    x_pixels: int,
) -> np.ndarray:
    """Build a (1, H, W) float32 intensity image from the current pixel_lists state."""
    counts = np.array([len(lst) for lst in pixel_lists], dtype=np.float32)
    counts = counts.reshape(y_pixels, total_x)[:, extra_left : extra_left + x_pixels]
    return counts[np.newaxis]


def poll_one_flim_frame(
    stream: object,
    x_pixels: int,
    y_pixels: int,
    extra_left: int,
    extra_right: int,
    pixel_dwell_ps: int,
    laser_ch: int,
    detector_ch: int,
    trigger_ch: int,
    final_pixel_margin_s: float = 1e-3,
    poll_sleep_s: float = 1e-4,
    progress_callback: Callable[[np.ndarray], None] | None = None,
) -> np.ndarray:
    """Poll a running TimeTagStream until one complete FLIM frame is collected.

    Waits for a single frame-start trigger on ``trigger_ch``, then bins all
    photon arrivals into pixels by time arithmetic.  Returns a
    ``(y_pixels, x_pixels)`` object array where each cell is an ``int64``
    array of photon arrival delays in picoseconds relative to the preceding
    laser pulse.
    """
    total_x = int(x_pixels) + int(extra_left) + int(extra_right)
    total_pixels = total_x * int(y_pixels)
    frame_duration_ps = total_pixels * int(pixel_dwell_ps)

    frame_start_ps: int | None = None
    frame_end_ps: int = 0
    deadline: float = 0.0
    all_laser_ts: list[np.ndarray] = []
    all_det_ts: list[np.ndarray] = []

    while True:
        data = stream.getData()  # type: ignore[attr-defined]

        if data.size == 0:
            if frame_start_ps is not None and time.monotonic() >= deadline:
                break
            time.sleep(poll_sleep_s)
            continue

        valid_mask = data.getEventTypes() == 0
        if not np.any(valid_mask):
            continue

        ts = data.getTimestamps()[valid_mask].astype(np.int64, copy=False)
        ch = data.getChannels()[valid_mask].astype(np.int64, copy=False)

        if frame_start_ps is None:
            trig_positions = np.flatnonzero(ch == trigger_ch)
            if not trig_positions.size:
                continue
            idx = int(trig_positions[0])
            frame_start_ps = int(ts[idx])
            frame_end_ps = frame_start_ps + frame_duration_ps
            deadline = time.monotonic() + frame_duration_ps / 1e12 + final_pixel_margin_s
            ts = ts[idx:]
            ch = ch[idx:]

        in_frame = ts <= frame_end_ps
        all_laser_ts.append(ts[(ch == laser_ch) & in_frame])
        all_det_ts.append(ts[(ch == detector_ch) & in_frame])

        if progress_callback is not None:
            laser_so_far = np.concatenate(all_laser_ts) if all_laser_ts else np.empty(0, dtype=np.int64)
            det_so_far = np.concatenate(all_det_ts) if all_det_ts else np.empty(0, dtype=np.int64)
            partial_lists: list[list[int]] = [[] for _ in range(total_pixels)]
            result = _compute_delays(det_so_far, laser_so_far)
            if result is not None:
                _bin_delays_into_pixels(
                    valid_det_ts=result[0],
                    delays_ps=result[1],
                    frame_start_ps=frame_start_ps,
                    pixel_dwell_ps=pixel_dwell_ps,
                    total_pixels=total_pixels,
                    pixel_lists=partial_lists,
                )
            progress_callback(_partial_intensity(partial_lists, y_pixels, total_x, extra_left, x_pixels))

        if np.any(ts > frame_end_ps) or time.monotonic() >= deadline:
            break

    if frame_start_ps is None:
        raise RuntimeError("Stream ended before the frame-start trigger was detected.")

    laser_ts = np.concatenate(all_laser_ts) if all_laser_ts else np.empty(0, dtype=np.int64)
    det_ts = np.concatenate(all_det_ts) if all_det_ts else np.empty(0, dtype=np.int64)

    pixel_lists: list[list[int]] = [[] for _ in range(total_pixels)]
    result = _compute_delays(det_ts, laser_ts)
    if result is not None:
        valid_det_ts, delays_ps = result
        _bin_delays_into_pixels(
            valid_det_ts=valid_det_ts,
            delays_ps=delays_ps,
            frame_start_ps=frame_start_ps,
            pixel_dwell_ps=pixel_dwell_ps,
            total_pixels=total_pixels,
            pixel_lists=pixel_lists,
        )

    return _finalize_frame(pixel_lists, y_pixels, total_x, extra_left, x_pixels)
