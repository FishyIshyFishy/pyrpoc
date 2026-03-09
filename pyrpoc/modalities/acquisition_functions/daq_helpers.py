from __future__ import annotations

from typing import Any

import numpy as np

from pyrpoc.instruments.confocal_daq import ConfocalDAQInstrument
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from pyrpoc.optocontrols.mask import MaskOptoControl


class DaqUnavailableError(RuntimeError):
    """Raised when a DAQ-backed acquisition cannot run on this machine."""


def extract_mask_contexts(
    opto_controls: list[tuple[BaseOptoControl, tuple[Any, ...]]],
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for control, payload in opto_controls:
        if not isinstance(control, MaskOptoControl):
            continue
        if control.mask_data is None:
            raise RuntimeError(f"Enabled mask control '{control.alias}' has no mask data")

        mask = np.asarray(control.mask_data, dtype=np.uint8)
        if mask.ndim != 2:
            raise RuntimeError(f"Mask control '{control.alias}' must provide a 2D mask")

        daq_port = int(control.daq_port)
        daq_line = int(control.daq_line)
        if len(payload) >= 2 and isinstance(payload[1], dict):
            daq_port = int(payload[1].get("daq_port", daq_port))
            daq_line = int(payload[1].get("daq_line", daq_line))

        if daq_port < 0 or daq_line < 0:
            raise RuntimeError(f"Mask control '{control.alias}' has invalid DAQ port/line values")

        contexts.append(
            {
                "alias": control.alias,
                "mask": mask,
                "daq_port": daq_port,
                "daq_line": daq_line,
            }
        )
    return contexts


def acquire_confocal_frame_with_daq(
    *,
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
    mask_contexts: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    return _acquire_with_nidaq(
        device_name=daq_instrument.device_name,
        sample_rate_hz=float(daq_instrument.sample_rate_hz),
        fast_axis_channel=int(daq_instrument.fast_axis_ao),
        slow_axis_channel=int(daq_instrument.slow_axis_ao),
        active_ai_channels=active_ai_channels,
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        extra_left=extra_left,
        extra_right=extra_right,
        dwell_time_us=dwell_time_us,
        fast_axis_offset=fast_axis_offset,
        fast_axis_amplitude=fast_axis_amplitude,
        slow_axis_offset=slow_axis_offset,
        slow_axis_amplitude=slow_axis_amplitude,
        mask_contexts=mask_contexts,
    )


def _acquire_with_nidaq(
    *,
    device_name: str,
    sample_rate_hz: float,
    fast_axis_channel: int,
    slow_axis_channel: int,
    active_ai_channels: list[int],
    x_pixels: int,
    y_pixels: int,
    extra_left: int,
    extra_right: int,
    dwell_time_us: float,
    fast_axis_offset: float,
    fast_axis_amplitude: float,
    slow_axis_offset: float,
    slow_axis_amplitude: float,
    mask_contexts: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    x_pixels = int(x_pixels)
    y_pixels = int(y_pixels)
    extra_left = max(0, int(extra_left))
    extra_right = max(0, int(extra_right))
    if x_pixels <= 0 or y_pixels <= 0:
        raise ValueError("Scan dimensions must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("Confocal DAQ sample rate must be positive")

    if not active_ai_channels:
        raise ValueError("No active AI channels configured")

    pixel_samples = max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))
    total_x = x_pixels + extra_left + extra_right
    total_y = y_pixels
    total_samples = total_x * total_y * pixel_samples
    if total_samples <= 0:
        raise ValueError("Invalid DAQ acquisition sample count")

    waveform = _build_raster_waveform(
        x_pixels=total_x,
        y_pixels=total_y,
        pixel_samples=pixel_samples,
        fast_axis_offset=fast_axis_offset,
        fast_axis_amplitude=fast_axis_amplitude,
        slow_axis_offset=slow_axis_offset,
        slow_axis_amplitude=slow_axis_amplitude,
    )
    ttl_signals = _build_mask_ttl_signals(
        mask_contexts=mask_contexts or [],
        total_x=total_x,
        total_y=total_y,
        pixel_samples=pixel_samples,
        extra_left=extra_left,
        device_name=device_name,
    )

    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType
    except Exception as exc:
        raise DaqUnavailableError(f"nidaqmx is unavailable ({exc})") from exc

    ai_channels = [f"{device_name}/ai{idx}" for idx in active_ai_channels]

    do_task: nidaqmx.Task | None = None
    static_do_task: nidaqmx.Task | None = None
    static_values: list[bool] = []

    try:
        with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
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
                    do_task = nidaqmx.Task()
                    for channel_name in dynamic_channels:
                        do_task.do_channels.add_do_chan(channel_name)
                    do_task.timing.cfg_samp_clk_timing(
                        rate=sample_rate_hz,
                        source=f"/{device_name}/ao/SampleClock",
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=total_samples,
                    )
                    if len(dynamic_channels) == 1:
                        do_task.write(dynamic_ttls[0].tolist(), auto_start=False)
                    else:
                        do_task.write([ttl.tolist() for ttl in dynamic_ttls], auto_start=False)

                if static_channels:
                    static_do_task = nidaqmx.Task()
                    for channel_name in static_channels:
                        static_do_task.do_channels.add_do_chan(channel_name)
                    static_do_task.write(static_values, auto_start=True)

            ao_task.write(waveform, auto_start=False)
            ai_task.start()
            if do_task is not None:
                do_task.start()
            ao_task.start()

            timeout = total_samples / sample_rate_hz + 5
            ao_task.wait_until_done(timeout=timeout)
            ai_task.wait_until_done(timeout=timeout)
            if do_task is not None:
                do_task.wait_until_done(timeout=timeout)

            acq_data = ai_task.read(number_of_samples_per_channel=total_samples)
            acq_data = np.asarray(acq_data, dtype=np.float32)
            if acq_data.ndim == 1:
                acq_data = acq_data[np.newaxis, :]
            elif acq_data.ndim != 2:
                raise RuntimeError("Unexpected NI-DAQ data shape")

            frame_channels: list[np.ndarray] = []
            for channel_data in acq_data:
                reshaped = np.asarray(channel_data, dtype=np.float32).reshape(total_y, total_x, pixel_samples)
                pixel_values = np.mean(reshaped, axis=2)
                frame_channels.append(pixel_values[:, extra_left:extra_left + x_pixels].astype(np.float32))

            return np.stack(frame_channels, axis=0).astype(np.float32, copy=False)
    except DaqUnavailableError:
        raise
    except Exception as exc:
        raise DaqUnavailableError(f"Error during NI-DAQ acquisition ({exc})") from exc
    finally:
        if do_task is not None:
            do_task.close()
        if static_do_task is not None:
            if static_values:
                try:
                    static_do_task.write([not value for value in static_values], auto_start=True)
                except Exception:
                    pass
            static_do_task.close()


def _build_mask_ttl_signals(
    mask_contexts: list[dict[str, Any]],
    *,
    total_x: int,
    total_y: int,
    pixel_samples: int,
    extra_left: int,
    device_name: str,
) -> dict[str, np.ndarray]:
    ttl_signals: dict[str, np.ndarray] = {}
    for index, context in enumerate(mask_contexts):
        alias = str(context.get("alias", f"Mask {index}"))
        mask = np.asarray(context.get("mask"), dtype=np.uint8)
        if mask.ndim != 2:
            raise RuntimeError(f"Mask context '{alias}' must be a 2D array")

        daq_port = int(context.get("daq_port", 0))
        daq_line = int(context.get("daq_line", 0))
        if daq_port < 0 or daq_line < 0:
            raise RuntimeError(f"Mask context '{alias}' has invalid DAQ port/line values")

        padded_mask = np.zeros((total_y, total_x), dtype=bool)
        if mask.size:
            source_h = min(mask.shape[0], total_y)
            source_w = min(mask.shape[1], max(0, total_x - extra_left))
            if source_h > 0 and source_w > 0:
                padded_mask[:source_h, extra_left : extra_left + source_w] = mask[:source_h, :source_w] > 0

        ttl = np.zeros((total_y, total_x, pixel_samples), dtype=bool)
        ttl[padded_mask] = True
        ttl_signals[f"{device_name}/port{daq_port}/line{daq_line}"] = ttl.reshape(-1)
    return ttl_signals


def _build_raster_waveform(
    *,
    x_pixels: int,
    y_pixels: int,
    pixel_samples: int,
    fast_axis_offset: float,
    fast_axis_amplitude: float,
    slow_axis_offset: float,
    slow_axis_amplitude: float,
) -> np.ndarray:
    fast_amp = max(float(fast_axis_amplitude), 1e-6)
    slow_amp = max(float(slow_axis_amplitude), 1e-6)

    fast_axis = (np.linspace(-1.0, 1.0, x_pixels, endpoint=False, dtype=np.float32) * fast_amp) + float(
        fast_axis_offset
    )
    slow_axis = (np.linspace(-1.0, 1.0, y_pixels, endpoint=False, dtype=np.float32) * slow_amp) + float(
        slow_axis_offset
    )

    fast_samples = np.repeat(fast_axis, pixel_samples)
    slow_samples = np.repeat(slow_axis, x_pixels * pixel_samples)
    fast_raster = np.tile(fast_samples, y_pixels)
    slow_raster = np.array(slow_samples, copy=True)

    return np.column_stack((fast_raster, slow_raster)).astype(np.float64)
