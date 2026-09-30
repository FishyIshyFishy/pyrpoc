"""Split confocal: the same scan, each pixel's samples split into two windows.

The waveform arithmetic and NI task setup are copies of ``confocal.py``'s and
must stay identical to them. What differs is at the bottom: a pixel's samples
split into t0 and t2 windows, the mask TTL gated to t0, and the raw samples
published as their own output.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import nidaqmx as nx
import numpy as np
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader

from pyrpoc.plugins.devices import DAQ, DaqError, Galvo
from pyrpoc.structs.data import Image2D, Samples4D
from pyrpoc.structs.program import Program, program_registry

from .components.param_groups import (
    DaqGroup,
    Mask,
    ModulationGroup,
    ScanGroup,
    SplitGroup,
)
from .components.runners import Continuous, Single


def pixel_samples(dwell_time_us: float, sample_rate_hz: float) -> int:
    """Samples per pixel. Truncating, floor 1 -- as confocal, not as FLIM."""
    return max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))


def generate_raster_waveform(
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
    """The (fast, slow) AO waveform for one frame: the fast axis sweeps the
    overscan-padded width each line, holding ``pixel_samples`` per pixel."""
    total_x = extra_left + x_pixels + extra_right
    fast_step = 2.0 * fast_axis_amplitude / x_pixels
    fast_start = -fast_axis_amplitude - extra_left * fast_step
    fast_axis = fast_start + np.arange(total_x, dtype=np.float32) * fast_step + fast_axis_offset
    slow_axis = (
        np.linspace(-1.0, 1.0, y_pixels, endpoint=False, dtype=np.float32) * slow_axis_amplitude
        + slow_axis_offset
    )
    fast_raster = np.tile(np.repeat(fast_axis, pixel_samples), y_pixels)
    slow_raster = np.repeat(slow_axis, total_x * pixel_samples)
    return np.vstack((fast_raster, slow_raster)).astype(np.float64)


def waveform_for_scan(scan: ScanGroup, pixel_samples: int) -> np.ndarray:
    return generate_raster_waveform(
        x_pixels=scan.x_pixels,
        extra_left=scan.extra_left,
        extra_right=scan.extra_right,
        y_pixels=scan.y_pixels,
        pixel_samples=pixel_samples,
        fast_axis_offset=scan.fast_axis_offset,
        fast_axis_amplitude=scan.fast_axis_amplitude,
        slow_axis_offset=scan.slow_axis_offset,
        slow_axis_amplitude=scan.slow_axis_amplitude,
    )


def extract_kept_samples(
    channel_data: np.ndarray,
    total_y: int,
    total_x: int,
    pixel_samples: int,
    extra_left: int,
    x_pixels: int,
) -> np.ndarray:
    """Drop the overscan columns from one channel's raw sample stream."""
    scan_line = np.asarray(channel_data, dtype=np.float32).reshape(total_y, total_x * pixel_samples)
    pixel_grid = scan_line.reshape(total_y, total_x, pixel_samples)
    kept = pixel_grid[:, extra_left : extra_left + x_pixels, :]
    return kept.reshape(total_y, x_pixels * pixel_samples).astype(np.float32, copy=False)


def resize_mask_nearest(mask_bool: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    source_h, source_w = mask_bool.shape
    y_idx = np.minimum((np.arange(target_h, dtype=np.int64) * source_h) // target_h, source_h - 1)
    x_idx = np.minimum((np.arange(target_w, dtype=np.int64) * source_w) // target_w, source_w - 1)
    return mask_bool[np.ix_(y_idx, x_idx)]


def mask_on_scan_grid(mask: np.ndarray, scan: ScanGroup) -> np.ndarray:
    """The mask resized onto the kept pixels, padded with the overscan columns."""
    mask_bool = np.asarray(mask, dtype=np.uint8) > 0
    if mask_bool.shape != (scan.y_pixels, scan.x_pixels):
        mask_bool = resize_mask_nearest(mask_bool, scan.y_pixels, scan.x_pixels)
    padded = np.zeros((scan.y_pixels, scan.total_x), dtype=bool)
    padded[:, scan.extra_left : scan.extra_left + scan.x_pixels] = mask_bool
    return padded


def mask_ttl(
    masks: Sequence[Mask],
    *,
    scan: ScanGroup,
    pixel_samples: int,
    device_name: str,
) -> dict[str, np.ndarray]:
    """One flat boolean TTL signal per bound digital line. A mask that is all
    zero on the scan grid gets no line, so no DO task is created for it."""
    ttl_signals: dict[str, np.ndarray] = {}
    for binding in masks:
        # Narrows for the type checker: a run's masks are resolved at start.
        if binding.array is None:
            raise ValueError(f"mask '{binding.describe()}' was not resolved")
        padded = mask_on_scan_grid(binding.array, scan)
        if not np.any(padded):
            continue
        ttl = np.zeros((scan.y_pixels, scan.total_x, pixel_samples), dtype=bool)
        ttl[padded] = True
        ttl_signals[binding.channel(device_name)] = ttl.reshape(-1)
    return ttl_signals


def split_mask_ttl(
    masks: Sequence[Mask],
    *,
    scan: ScanGroup,
    pixel_samples: int,
    device_name: str,
    t0_samples: int,
) -> dict[str, np.ndarray]:
    """``mask_ttl`` truncated to the first ``t0_samples`` of every pixel."""
    signals = mask_ttl(masks, scan=scan, pixel_samples=pixel_samples, device_name=device_name)
    if t0_samples >= pixel_samples:
        return signals
    for signal in signals.values():
        signal.reshape(-1, pixel_samples)[:, t0_samples:] = False
    return signals


def run_raster(
    device_name: str,
    sample_rate_hz: float,
    fast_ao: int,
    slow_ao: int,
    waveform: np.ndarray,
    ttl_signals: dict[str, np.ndarray],
    x_pixels: int,
    y_pixels: int,
    extra_left: int,
    extra_right: int,
    dwell_time_us: float,
    ai_channels: list[int],
) -> tuple[np.ndarray, int, int, int]:
    """Drive AO, read AI and clock DO as one synchronised finite acquisition."""
    fast_axis_channel = int(fast_ao)
    slow_axis_channel = int(slow_ao)

    samples_per_pixel = pixel_samples(dwell_time_us, sample_rate_hz)
    total_x = x_pixels + extra_left + extra_right
    total_y = y_pixels
    total_samples = total_x * total_y * samples_per_pixel

    ai_channel_names = [f"{device_name}/ai{idx}" for idx in ai_channels]
    do_task: nx.Task | None = None
    static_do_task: nx.Task | None = None
    static_values: list[bool] = []

    try:
        with nx.Task() as ao_task, nx.Task() as ai_task:
            ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_axis_channel}")
            ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_axis_channel}")
            for ch in ai_channel_names:
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
                    payload = (
                        dynamic_ttls[0].tolist()
                        if len(dynamic_channels) == 1
                        else [t.tolist() for t in dynamic_ttls]
                    )
                    # Many samples per line, so write() does not auto-start.
                    do_task.write(payload)

                if static_channels:
                    static_do_task = nx.Task()
                    for ch in static_channels:
                        static_do_task.do_channels.add_do_chan(ch)
                    # One sample per line, so write() auto-starts.
                    static_do_task.write(static_values)

            ao_task.write(np.asarray(waveform, dtype=np.float64))
            ai_task.start()
            if do_task is not None:
                do_task.start()
            ao_task.start()

            timeout = total_samples / sample_rate_hz + 5
            ao_task.wait_until_done(timeout=timeout)
            ai_task.wait_until_done(timeout=timeout)
            if do_task is not None:
                do_task.wait_until_done(timeout=timeout)

            # Task.read is typed to accept no sample count; the stream reader
            # takes one, and always fills (channels, samples).
            raw = np.empty((ai_task.number_of_channels, total_samples), dtype=np.float64)
            AnalogMultiChannelReader(ai_task.in_stream).read_many_sample(
                raw, number_of_samples_per_channel=total_samples
            )
            acq_data = raw.astype(np.float32)

            channels_out = [
                extract_kept_samples(
                    ch_data, total_y, total_x, samples_per_pixel, extra_left, x_pixels
                )
                for ch_data in acq_data
            ]
            return (
                np.stack(channels_out, axis=0).astype(np.float32, copy=False),
                total_y,
                x_pixels,
                samples_per_pixel,
            )

    except Exception as exc:
        raise DaqError(f"NI-DAQ acquisition failed: {exc}") from exc
    finally:
        if do_task is not None:
            do_task.close()
        if static_do_task is not None:
            if static_values:
                with contextlib.suppress(Exception):
                    static_do_task.write([not v for v in static_values])
            static_do_task.close()


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
    split_point = t0_samples
    second_start = t0_samples + t1_samples

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
    daq: DAQ,
    galvo: Galvo,
    scan: ScanGroup,
    sample_rate_hz: float,
    split: SplitGroup,
    ttl: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """One split-confocal raster scan. Returns ``(split_frame, raw_frame)``."""
    samples_per_pixel = pixel_samples(scan.dwell_time_us, sample_rate_hz)

    scan_data, total_y_out, x_out, px_out = run_raster(
        device_name=daq.config.device_name,
        sample_rate_hz=sample_rate_hz,
        fast_ao=galvo.config.fast_ao,
        slow_ao=galvo.config.slow_ao,
        waveform=waveform_for_scan(scan, samples_per_pixel),
        ttl_signals=ttl,
        x_pixels=scan.x_pixels,
        y_pixels=scan.y_pixels,
        extra_left=scan.extra_left,
        extra_right=scan.extra_right,
        dwell_time_us=scan.dwell_time_us,
        ai_channels=list(daq.config.ai_channels),
    )
    return reshape_to_split_frame(
        scan_data, total_y_out, x_out, px_out, split.t0_samples, split.t1_samples
    )


def channel_labels(daq: DAQ) -> list[str]:
    """``ai0_t0``, ``ai0_t2``, ``ai1_t0``, ...: interleaved, matching the frame."""
    labels: list[str] = []
    for index in daq.config.ai_channels:
        labels.append(f"ai{index}_t0")
        labels.append(f"ai{index}_t2")
    return labels


def build_ttl(
    scan: ScanGroup,
    modulation: ModulationGroup,
    daq_params: DaqGroup,
    split: SplitGroup,
    daq: DAQ,
) -> dict[str, np.ndarray]:
    """Mask TTL gated to the first ``t0_samples`` of every pixel."""
    return split_mask_ttl(
        modulation.masks,
        scan=scan,
        pixel_samples=pixel_samples(scan.dwell_time_us, daq_params.sample_rate_hz),
        device_name=daq.config.device_name,
        t0_samples=split.t0_samples,
    )


@program_registry.register("split_confocal")
class SplitConfocal(Program):
    display_name = "Split Confocal"
    uses = [Galvo, DAQ]
    params = [ScanGroup, DaqGroup, SplitGroup, ModulationGroup]
    emits = {"intensity": Image2D, "raw_pixel_stream": Samples4D}
    runners = [Single(), Continuous()]

    def run(self, ctx) -> None:
        scan = ctx.params[ScanGroup]
        daq_params = ctx.params[DaqGroup]
        split = ctx.params[SplitGroup]
        modulation = ctx.params[ModulationGroup]
        num_frames = scan.num_frames

        daq: DAQ = ctx.devices[DAQ]
        galvo: Galvo = ctx.devices[Galvo]

        ttl = build_ttl(scan, modulation, daq_params, split, daq)
        labels = channel_labels(daq)

        for index in range(num_frames):
            ctx.check_cancel()
            ctx.status(f"frame {index + 1}/{num_frames}")
            split_frame, raw = split_raster_scan(
                daq=daq,
                galvo=galvo,
                scan=scan,
                sample_rate_hz=daq_params.sample_rate_hz,
                split=split,
                ttl=ttl,
            )
            ctx.publish("intensity", split_frame, channels=labels)
            ctx.publish("raw_pixel_stream", raw)
