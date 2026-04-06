from __future__ import annotations

import os

import numpy as np

import nidaqmx as nx
from nidaqmx.constants import AcquisitionType

from pyrpoc.backend_utils.opto_control_contexts import MaskContext
from pyrpoc.instruments.confocal_daq import ConfocalDAQInstrument
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from pyrpoc.optocontrols.mask import MaskOptoControl

_DAQ_MASK_DEBUG = os.getenv("PYRPOC_DAQ_MASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


class DaqUnavailableError(RuntimeError):
    """Raised when a DAQ-backed acquisition cannot run on this machine."""


# TODO: make the need for a context object uniform to all optocontrol types
# so that it is easier to know in different modalities what the mask info is shipped as
# this should be enforced by BaseOptoControl
# and we can have a Context object
# this might be a good way to ship DisplayContexts and InstrumentContexts as well

# TODO: make sure that everything needed for acquisition parameter is read into the acquisition object
# and verified at a specific layer enforced by base
# so that we dont need a bunch of validation logic in acquisition code

# TODO: make sure that saving of things is done in acquisition layer
# so that modality service stays agnostic

def extract_mask_contexts(opto_controls: list[BaseOptoControl]) -> list[MaskContext]:
    contexts: list[MaskContext] = []
    for control in opto_controls:
        if not isinstance(control, MaskOptoControl):
            continue

        context = control.context
        if not isinstance(context, MaskContext):
            raise TypeError(f"{type(control).__name__} must prepare a MaskContext before acquisition")

        contexts.append(context)
    return contexts


def generate_raster_waveform(
    total_x: int,
    x_pixels: int,
    extra_left: int,
    extra_right: int,
    y_pixels: int,
    pixel_samples: int,
    fast_axis_offset: float,
    fast_axis_amplitude: float,
    slow_axis_offset: float,
    slow_axis_amplitude: float,
) -> np.ndarray:
    fast_amp = max(float(fast_axis_amplitude), 1e-6)
    slow_amp = max(float(slow_axis_amplitude), 1e-6)

    if x_pixels <= 0:
        raise ValueError("x_pixels must be positive to generate raster waveform")
    if total_x != (x_pixels + extra_left + extra_right):
        raise ValueError("total_x must equal x_pixels + extra_left + extra_right")

    # Compute fast-axis step from the in-FOV pixel count, then extrapolate for extras.
    fast_step = (2.0 * fast_amp) / float(x_pixels)
    fast_start = -fast_amp - (float(extra_left) * fast_step)
    fast_axis = fast_start + (np.arange(total_x, dtype=np.float32) * fast_step)
    fast_axis = fast_axis + float(fast_axis_offset)
    slow_axis = (np.linspace(-1.0, 1.0, y_pixels, endpoint=False, dtype=np.float32) * slow_amp) + float(slow_axis_offset)

    fast_samples = np.repeat(fast_axis, pixel_samples)
    slow_samples = np.repeat(slow_axis, total_x * pixel_samples)
    fast_raster = np.tile(fast_samples, y_pixels)
    slow_raster = np.tile(slow_samples, 1)

    return np.vstack((fast_raster, slow_raster)).astype(np.float64)


def _resize_mask_nearest(mask_bool: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if target_h <= 0 or target_w <= 0:
        raise ValueError("Target mask size must be positive")
    source_h, source_w = mask_bool.shape
    if source_h <= 0 or source_w <= 0:
        return np.zeros((target_h, target_w), dtype=bool)

    y_idx = np.minimum((np.arange(target_h, dtype=np.int64) * source_h) // target_h, source_h - 1)
    x_idx = np.minimum((np.arange(target_w, dtype=np.int64) * source_w) // target_w, source_w - 1)
    return mask_bool[np.ix_(y_idx, x_idx)]


def _preprocess_mask_to_scan_grid(
    raw_mask: object,
    *,
    total_x: int,
    total_y: int,
    scan_x_pixels: int,
    extra_left: int,
    extra_right: int,
) -> np.ndarray:
    if total_x <= 0 or total_y <= 0:
        raise ValueError("total_x and total_y must be positive")
    if scan_x_pixels < 0 or extra_left < 0 or extra_right < 0:
        raise ValueError("scan_x_pixels and extra pixels must be non-negative")
    if (extra_left + scan_x_pixels + extra_right) != total_x:
        raise ValueError(
            "Final mask payload width mismatch: expected total_x == extra_left + scan_x_pixels + extra_right"
        )

    mask = np.asarray(raw_mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape={mask.shape}")

    if scan_x_pixels == 0:
        return np.zeros((total_y, total_x), dtype=bool)

    mask_bool = mask > 0
    if mask_bool.shape != (total_y, scan_x_pixels):
        mask_bool = _resize_mask_nearest(mask_bool, target_h=total_y, target_w=scan_x_pixels)

    if mask_bool.shape[1] != scan_x_pixels:
        raise RuntimeError(f"Processed mask width {mask_bool.shape[1]} does not match scan_x_pixels={scan_x_pixels}")

    padded_mask = np.zeros((total_y, total_x), dtype=bool)
    padded_mask[:, extra_left : extra_left + scan_x_pixels] = mask_bool
    return padded_mask


def _format_mask_bbox(mask_2d: np.ndarray) -> str:
    if not np.any(mask_2d):
        return "none"
    ys, xs = np.nonzero(mask_2d)
    return f"x=[{int(xs.min())},{int(xs.max())}], y=[{int(ys.min())},{int(ys.max())}]"


def _emit_mask_debug(prefix: str, channel_name: str, mask_2d: np.ndarray, ttl_3d: np.ndarray) -> None:
    active_pixels = int(np.count_nonzero(mask_2d))
    total_pixels = int(mask_2d.size)
    pct = (100.0 * active_pixels / total_pixels) if total_pixels else 0.0
    active_ttl_samples = int(np.count_nonzero(ttl_3d))
    bbox = _format_mask_bbox(mask_2d)
    print(
        f"[DAQ mask debug] {prefix} {channel_name}: "
        f"active={active_pixels}/{total_pixels} ({pct:.2f}%), bbox={bbox}, active_ttl_samples={active_ttl_samples}"
    )


def _extract_kept_samples(
    channel_data: np.ndarray,
    *,
    total_y: int,
    total_x: int,
    pixel_samples: int,
    extra_left: int,
    x_pixels: int,
) -> np.ndarray:
    row_samples = total_x * pixel_samples
    scan_line_samples = np.asarray(channel_data, dtype=np.float32).reshape(total_y, row_samples)
    pixel_grid = scan_line_samples.reshape(total_y, total_x, pixel_samples)
    kept_pixels = pixel_grid[:, extra_left : extra_left + x_pixels, :]
    if kept_pixels.shape[1] != x_pixels:
        raise RuntimeError("Acquired scan payload does not match requested scan dimensions")
    return kept_pixels.reshape(total_y, x_pixels * pixel_samples).astype(np.float32, copy=False)


def generate_mask_ttl_signals(
    *,
    total_x: int,
    total_y: int,
    pixel_samples: int,
    extra_left: int,
    extra_right: int,
    device_name: str,
    mask_contexts: list[MaskContext],
    scan_x_pixels: int,
    debug: bool = False,
) -> dict[str, np.ndarray]:
    ttl_signals: dict[str, np.ndarray] = {}

    for context in mask_contexts:
        if context.mask is None:
            continue

        channel_name = f"{device_name}/port{int(context.daq_port)}/line{int(context.daq_line)}"
        try:
            padded_mask = _preprocess_mask_to_scan_grid(
                context.mask,
                total_x=total_x,
                total_y=total_y,
                scan_x_pixels=scan_x_pixels,
                extra_left=extra_left,
                extra_right=extra_right,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to preprocess mask for {channel_name}: {exc}") from exc

        if not np.any(padded_mask):
            continue

        ttl = np.zeros((total_y, total_x, pixel_samples), dtype=bool)
        ttl[padded_mask] = True
        ttl_signals[channel_name] = ttl.reshape(-1)
        if debug:
            _emit_mask_debug("full", channel_name, padded_mask, ttl)
    return ttl_signals


def generate_mask_ttl_signals_split(
    *,
    total_x: int,
    total_y: int,
    pixel_samples: int,
    extra_left: int,
    extra_right: int,
    device_name: str,
    mask_contexts: list[MaskContext],
    scan_x_pixels: int,
    t0_samples: int,
    debug: bool = False,
) -> dict[str, np.ndarray]:
    ttl_signals = generate_mask_ttl_signals(
        total_x=total_x,
        total_y=total_y,
        pixel_samples=pixel_samples,
        extra_left=extra_left,
        extra_right=extra_right,
        device_name=device_name,
        mask_contexts=mask_contexts,
        scan_x_pixels=scan_x_pixels,
        debug=debug,
    )

    if t0_samples >= pixel_samples:
        return ttl_signals

    for channel_name, ttl in ttl_signals.items():
        ttl = ttl.reshape(total_y, total_x, pixel_samples)
        ttl[:, :, t0_samples:] = False
        ttl_signals[channel_name] = ttl.reshape(-1)
        if debug:
            _emit_mask_debug("split", channel_name, ttl.any(axis=2), ttl)
    return ttl_signals


def acquire_daq(
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
    device_name: str = daq_instrument.device_name
    sample_rate_hz: float = float(daq_instrument.sample_rate_hz)
    fast_axis_channel: int = int(daq_instrument.fast_axis_ao)
    slow_axis_channel: int = int(daq_instrument.slow_axis_ao)

    pixel_samples = max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))
    total_x = x_pixels + extra_left + extra_right
    total_y = y_pixels
    total_samples = total_x * total_y * pixel_samples

    waveform_array = np.asarray(waveform, dtype=np.float64)
    ai_channels = [f"{device_name}/ai{idx}" for idx in active_ai_channels]

    do_task: nx.Task | None = None
    static_do_task: nx.Task | None = None
    static_values: list[bool] = []

    try:
        with nx.Task() as ao_task, nx.Task() as ai_task:
            ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_axis_channel}")
            ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_axis_channel}")
            for channel in ai_channels:
                ai_task.ai_channels.add_ai_voltage_chan(channel)

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
                dynamic_channels: list[str] = []
                dynamic_ttls: list[np.ndarray] = []
                static_channels: list[str] = []
                static_values = []

                for channel_name, ttl in ttl_signals.items():
                    if "/port0/" in channel_name.lower():
                        dynamic_channels.append(channel_name)
                        dynamic_ttls.append(ttl)
                    else:
                        static_channels.append(channel_name)
                        static_values.append(bool(ttl.flat[0]))

                if dynamic_channels:
                    do_task = nx.Task()
                    for channel_name in dynamic_channels:
                        do_task.do_channels.add_do_chan(channel_name)
                    do_task.timing.cfg_samp_clk_timing(
                        rate=sample_rate_hz,
                        source=f"/{device_name}/ao/SampleClock",
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=total_samples,
                    )
                    if len(dynamic_channels) == 1:
                        do_task.write(dynamic_ttls[0].tolist(), auto_start=False)  # pyright:ignore
                    else:
                        do_task.write([ttl.tolist() for ttl in dynamic_ttls], auto_start=False)  # pyright:ignore

                if static_channels:
                    static_do_task = nx.Task()
                    for channel_name in static_channels:
                        static_do_task.do_channels.add_do_chan(channel_name)
                    static_do_task.write(static_values, auto_start=True)  # pyright:ignore

            ao_task.write(waveform_array, auto_start=False)  # pyright:ignore
            ai_task.start()
            if do_task is not None:
                do_task.start()
            ao_task.start()

            timeout = total_samples / sample_rate_hz + 5
            ao_task.wait_until_done(timeout=timeout)
            ai_task.wait_until_done(timeout=timeout)
            if do_task is not None:
                do_task.wait_until_done(timeout=timeout)

            acq_data = ai_task.read(number_of_samples_per_channel=total_samples)  # pyright:ignore
            acq_data = np.asarray(acq_data, dtype=np.float32)
            if acq_data.ndim == 1:
                acq_data = acq_data[np.newaxis, :]
            elif acq_data.ndim != 2:
                raise RuntimeError("Unexpected NI-DAQ data shape")

            frame_channels: list[np.ndarray] = []
            for channel_data in acq_data:
                kept_samples = _extract_kept_samples(
                    channel_data,
                    total_y=total_y,
                    total_x=total_x,
                    pixel_samples=pixel_samples,
                    extra_left=extra_left,
                    x_pixels=x_pixels,
                )
                frame_channels.append(kept_samples.astype(np.float32, copy=False))

            return np.stack(frame_channels, axis=0).astype(np.float32, copy=False), total_y, x_pixels, pixel_samples
    except Exception as exc:
        print(f"Error during NI-DAQ acquisition ({exc})")
        raise DaqUnavailableError(f"Error during NI-DAQ acquisition ({exc})") from exc
    finally:
        if do_task is not None:
            do_task.close()
        if static_do_task is not None:
            if static_values:
                try:
                    static_do_task.write([not value for value in static_values], auto_start=True)  # pyright:ignore
                except Exception:
                    pass
            static_do_task.close()


def reshape_data(
    scan_data: np.ndarray,
    total_y: int,
    x_pixels: int,
    pixel_samples: int,
) -> np.ndarray:
    frame_channels: list[np.ndarray] = []
    for channel_data in scan_data:
        channel_pixels = np.asarray(channel_data, dtype=np.float32).reshape(total_y, x_pixels, pixel_samples)
        frame_channels.append(channel_pixels.mean(axis=2))
    return np.stack(frame_channels, axis=0).astype(np.float32, copy=False)


def reshape_data_split(
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

    for channel_data in scan_data:
        pixel_data = np.asarray(channel_data, dtype=np.float32).reshape(total_y, x_pixels, pixel_samples)
        raw_channels.append(pixel_data.astype(np.float32, copy=False))

        first_portion = pixel_data[:, :, :split_point].mean(axis=2)
        if second_start < pixel_samples:
            second_portion = pixel_data[:, :, second_start:].mean(axis=2)
        else:
            second_portion = np.zeros_like(first_portion)

        split_channels.append(first_portion.astype(np.float32, copy=False))
        split_channels.append(second_portion.astype(np.float32, copy=False))

    return (
        np.stack(split_channels, axis=0).astype(np.float32, copy=False),
        np.stack(raw_channels, axis=0).astype(np.float32, copy=False),
    )


def acquire_daq_confocal(
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
    mask_contexts: list[MaskContext],
) -> np.ndarray:
    sample_rate_hz: float = float(daq_instrument.sample_rate_hz)
    pixel_samples = max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))
    total_x = x_pixels + extra_left + extra_right

    waveform = generate_raster_waveform(
        total_x=total_x,
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
    ttl_signals = generate_mask_ttl_signals(
        total_x=total_x,
        total_y=y_pixels,
        pixel_samples=pixel_samples,
        extra_left=extra_left,
        extra_right=extra_right,
        device_name=daq_instrument.device_name,
        mask_contexts=mask_contexts,
        scan_x_pixels=x_pixels,
        debug=_DAQ_MASK_DEBUG,
    )
    scan_data, total_y, x_pixels_scanned, scan_pixel_samples = acquire_daq(
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
    return reshape_data(scan_data, total_y, x_pixels_scanned, scan_pixel_samples)


def acquire_daq_flim(
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
    mask_contexts: list[MaskContext],
    pixel_clock_do_line: int,
) -> np.ndarray:
    """Like acquire_daq_confocal but also outputs a pixel clock on port0.

    A single TTL pulse (one sample wide) is emitted at the start of each pixel
    on the specified port0 line so that the TimeTagger can synchronise photon
    arrivals to scan pixels.
    """
    from .flim_helpers import generate_pixel_clock_signal

    sample_rate_hz: float = float(daq_instrument.sample_rate_hz)
    pixel_samples = max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))
    total_x = x_pixels + extra_left + extra_right

    waveform = generate_raster_waveform(
        total_x=total_x,
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
    ttl_signals = generate_mask_ttl_signals(
        total_x=total_x,
        total_y=y_pixels,
        pixel_samples=pixel_samples,
        extra_left=extra_left,
        extra_right=extra_right,
        device_name=daq_instrument.device_name,
        mask_contexts=mask_contexts,
        scan_x_pixels=x_pixels,
        debug=_DAQ_MASK_DEBUG,
    )
    pixel_clock_chan = f"{daq_instrument.device_name}/port0/line{int(pixel_clock_do_line)}"
    ttl_signals[pixel_clock_chan] = generate_pixel_clock_signal(total_x, y_pixels, pixel_samples)

    scan_data, total_y, x_pixels_scanned, scan_pixel_samples = acquire_daq(
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
    return reshape_data(scan_data, total_y, x_pixels_scanned, scan_pixel_samples)


def acquire_daq_split_confocal(
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
    sample_rate_hz: float = float(daq_instrument.sample_rate_hz)
    pixel_samples = max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))
    total_x = x_pixels + extra_left + extra_right

    waveform = generate_raster_waveform(
        total_x=total_x,
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
    ttl_signals = generate_mask_ttl_signals_split(
        total_x=total_x,
        total_y=y_pixels,
        pixel_samples=pixel_samples,
        extra_left=extra_left,
        extra_right=extra_right,
        device_name=daq_instrument.device_name,
        mask_contexts=mask_contexts,
        scan_x_pixels=x_pixels,
        t0_samples=t0_samples,
        debug=_DAQ_MASK_DEBUG,
    )
    scan_data, total_y, x_pixels_scanned, scan_pixel_samples = acquire_daq(
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
    return reshape_data_split(
        scan_data=scan_data,
        total_y=total_y,
        x_pixels=x_pixels_scanned,
        pixel_samples=scan_pixel_samples,
        t0_samples=t0_samples,
        t1_samples=t1_samples,
    )
