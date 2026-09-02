"""FLIM: the galvo scan that emits tagger markers, and reading a frame back.

Moved from ``modalities/flim/acquisition_core.py``. Carried over line for line;
the counter-derived pixel clock and the exported start trigger only fail on the
instrument and nothing here can catch a change to them.
"""

from __future__ import annotations

import numpy as np
import nidaqmx as nx
from nidaqmx.constants import AcquisitionType, Signal

from pyrpoc.core.errors import DaqError

from .raster import generate_raster_waveform


def pixel_samples(dwell_time_us: float, sample_rate_hz: float) -> int:
    """Samples per pixel for a FLIM scan.

    Rounds, floor 2 — the counter needs at least one high tick and one low tick.
    The raster path truncates and has a floor of 1; the two formulas differ and
    ``tests/operations/test_pixel_samples.py`` pins both. Do not unify them.
    """
    return max(2, int(round(dwell_time_us * 1e-6 * sample_rate_hz)))


def run_flim_scan(
    device_name: str,
    sample_rate_hz: float,
    fast_ao: int,
    slow_ao: int,
    raster_waveform: np.ndarray,
    n_pixels: int,
    pixel_samples: int,
    frame_trigger_pfi: int,
    pixel_clock_ctr: int,
    pixel_clock_pfi: int,
) -> None:
    """Drive one galvo raster while emitting the two markers the TimeTagger
    needs: a frame-start trigger (the exported AO start trigger) and a pixel
    clock (a counter pulse every ``pixel_samples`` AO sample-clock ticks).

    The pixel clock is divided down from the AO sample clock, so pixel
    boundaries stay locked to galvo position with no drift. No analog input is
    read — the FLIM image and lifetimes come from the photon stream.
    """
    total_samples = int(raster_waveform.shape[1])
    timeout = total_samples / sample_rate_hz + 5.0
    try:
        with nx.Task() as ao_task, nx.Task() as co_task:
            ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{int(fast_ao)}")
            ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{int(slow_ao)}")
            ao_task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=total_samples,
            )
            ao_task.export_signals.export_signal(
                Signal.START_TRIGGER, f"/{device_name}/PFI{int(frame_trigger_pfi)}"
            )

            co_channel = co_task.co_channels.add_co_pulse_chan_ticks(
                f"{device_name}/ctr{int(pixel_clock_ctr)}",
                source_terminal=f"/{device_name}/ao/SampleClock",
                high_ticks=1,
                low_ticks=int(pixel_samples) - 1,
            )
            co_channel.co_pulse_term = f"/{device_name}/PFI{int(pixel_clock_pfi)}"
            co_task.timing.cfg_implicit_timing(
                sample_mode=AcquisitionType.FINITE, samps_per_chan=int(n_pixels)
            )
            co_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                f"/{device_name}/ao/StartTrigger"
            )

            ao_task.write(np.asarray(raster_waveform, dtype=np.float64), auto_start=False)  # pyright:ignore
            co_task.start()   # arms and waits for the AO start trigger
            ao_task.start()
            ao_task.wait_until_done(timeout=timeout)
            co_task.wait_until_done(timeout=timeout)
    except Exception as exc:
        raise DaqError(f"NI-DAQ FLIM scan failed: {exc}") from exc


def flim_scan(
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
    fast_ao: int,
    slow_ao: int,
    frame_trigger_pfi: int,
    pixel_clock_ctr: int,
    pixel_clock_pfi: int,
    ai_channels: tuple[int, ...] | list[int] = (),
) -> None:
    """Build the raster waveform and run one FLIM scan.

    ``ai_channels`` is accepted and ignored: FLIM reads no analog input, but the
    DAQ device's configuration unpacks into this call alongside the galvo's.
    """
    del ai_channels
    samples_per_pixel = pixel_samples(dwell_time_us, sample_rate_hz)
    total_x = x_pixels + extra_left + extra_right
    n_pixels = total_x * y_pixels

    raster_waveform = generate_raster_waveform(
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
    run_flim_scan(
        device_name=device_name,
        sample_rate_hz=sample_rate_hz,
        fast_ao=fast_ao,
        slow_ao=slow_ao,
        raster_waveform=raster_waveform,
        n_pixels=n_pixels,
        pixel_samples=samples_per_pixel,
        frame_trigger_pfi=frame_trigger_pfi,
        pixel_clock_ctr=pixel_clock_ctr,
        pixel_clock_pfi=pixel_clock_pfi,
    )


def reshape_flim_frame(
    histograms: np.ndarray,
    n_bins: int,
    y_pixels: int,
    total_x_pixels: int,
    extra_left: int,
    x_pixels: int,
) -> np.ndarray:
    """Fold the flat ``(n_pixels, n_bins)`` Flim histogram into a
    ``(y_pixels, x_pixels, n_bins)`` float32 cube with the overscan columns
    clipped off."""
    cube = np.asarray(histograms, dtype=np.float32).reshape(y_pixels, total_x_pixels, n_bins)
    return cube[:, extra_left : extra_left + x_pixels, :]


def flim_intensity(hist_frame: np.ndarray) -> np.ndarray:
    """Collapse a ``(H, W, n_bins)`` histogram cube to a ``(H, W)`` photon-count
    intensity image."""
    return np.asarray(hist_frame, dtype=np.float32).sum(axis=2)


def read_flim_frame(
    flim_measurement,
    n_bins: int,
    y_pixels: int,
    total_x_pixels: int,
    extra_left: int,
    x_pixels: int,
) -> np.ndarray:
    """Read the current (just-scanned) Flim frame and return its clipped
    ``(y_pixels, x_pixels, n_bins)`` histogram cube."""
    frame = flim_measurement.getCurrentFrameEx()
    return reshape_flim_frame(
        frame.getHistograms(), n_bins, y_pixels, total_x_pixels, extra_left, x_pixels
    )
