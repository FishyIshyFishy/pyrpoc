"""Confocal: raster the galvo, read the analog inputs, publish a frame.

The program owns its loop and its hardware. There is no base class supplying
``while not should_stop``, no saving (the runner reads ``emits`` and creates
datasets with a save policy before ``run`` is called), and no frame counting.

The scan code lives here rather than in a shared operations folder: it is
acquisition-specific, and keeping it beside the program that runs it means
deleting this file deletes the whole modality. This is the canonical copy of
the waveform arithmetic; ``split_confocal.py`` and ``flim.py`` carry the same
functions verbatim and are meant to stay that way.

The NI task setup is not pinned by anything and only fails on the instrument,
so it is carried over line for line: the lowercase ``/port0/`` test that
separates clocked DO lines from static ones, the single-channel-versus-list
payload shape, the AI-then-DO-then-AO start ordering, the timeout, and the
inverted static write on teardown.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import nidaqmx as nx
from nidaqmx.constants import AcquisitionType

from pyrpoc.core.errors import DaqError
from pyrpoc.core.streams import Image2D
from pyrpoc.data.dataset import Dataset  # noqa: F401  (documents what publish writes into)
from pyrpoc.devices import DAQ, Galvo
from pyrpoc.run.program import Program

from .components import (
    DaqGroup,
    FramesGroup,
    Mask,
    ModulationGroup,
    ScanGroup,
)
from .registry import program_registry


# --------------------------------------------------------------------------- #
# Waveform arithmetic                                                          #
# --------------------------------------------------------------------------- #


def pixel_samples(dwell_time_us: float, sample_rate_hz: float) -> int:
    """Samples per pixel for a raster scan.

    Truncating, floor 1. The FLIM path rounds and has a floor of 2 because its
    counter needs a high tick and a low tick. The two formulas genuinely differ
    -- do not unify them.
    """
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
    '''
    create a waveform from the raster scan
    1. compute total pixels given the extra left and right
    2. compute total amplitude per line by padding the voltage step size to the left and right
        of the offset-amp and offset+amp points
    3. create waveforms
    '''
    total_x = extra_left + x_pixels + extra_right
    fast_amp = max(float(fast_axis_amplitude), 1e-6)
    slow_amp = max(float(slow_axis_amplitude), 1e-6)
    fast_step = (2.0 * fast_amp) / float(x_pixels)
    fast_start = -fast_amp - (float(extra_left) * fast_step)
    fast_axis = fast_start + (np.arange(total_x, dtype=np.float32) * fast_step) + float(fast_axis_offset)
    slow_axis = (
        np.linspace(-1.0, 1.0, y_pixels, endpoint=False, dtype=np.float32) * slow_amp
        + float(slow_axis_offset)
    )
    fast_raster = np.tile(np.repeat(fast_axis, pixel_samples), y_pixels)
    slow_raster = np.repeat(slow_axis, total_x * pixel_samples)
    return np.vstack((fast_raster, slow_raster)).astype(np.float64)


def waveform_for_scan(scan: ScanGroup, pixel_samples: int) -> np.ndarray:
    """``generate_raster_waveform`` driven from a ``ScanGroup``.

    ``generate_raster_waveform`` keeps its explicit nine-argument signature;
    this is the caller's side of it.
    """
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


def reshape_to_frame(
    scan_data: np.ndarray,
    total_y: int,
    x_pixels: int,
    pixel_samples: int,
) -> np.ndarray:
    """Mean over each pixel's samples, giving a ``(C, H, W)`` frame."""
    frame_channels = [
        np.asarray(ch, dtype=np.float32).reshape(total_y, x_pixels, pixel_samples).mean(axis=2)
        for ch in scan_data
    ]
    return np.stack(frame_channels, axis=0).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Masks to per-pixel TTL waveforms                                             #
# --------------------------------------------------------------------------- #


def resize_mask_nearest(mask_bool: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    source_h, source_w = mask_bool.shape
    if source_h <= 0 or source_w <= 0:
        return np.zeros((target_h, target_w), dtype=bool)
    y_idx = np.minimum((np.arange(target_h, dtype=np.int64) * source_h) // target_h, source_h - 1)
    x_idx = np.minimum((np.arange(target_w, dtype=np.int64) * source_w) // target_w, source_w - 1)
    return mask_bool[np.ix_(y_idx, x_idx)]


def preprocess_mask_to_scan_grid(
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
        mask_bool = resize_mask_nearest(mask_bool, target_h=total_y, target_w=scan_x_pixels)

    padded = np.zeros((total_y, total_x), dtype=bool)
    padded[:, extra_left : extra_left + scan_x_pixels] = mask_bool
    return padded


def mask_ttl(
    masks: Sequence[tuple[Mask, np.ndarray]],
    *,
    scan: ScanGroup,
    pixel_samples: int,
    device_name: str,
) -> dict[str, np.ndarray]:
    """One flat boolean TTL signal per bound digital line.

    A mask that is entirely zero after padding is skipped, so its channel does
    not appear in the result and no DO task is created for it.
    """
    total_x = scan.total_x
    total_y = scan.y_pixels

    ttl_signals: dict[str, np.ndarray] = {}
    for binding, mask in masks:
        if mask is None:
            continue
        channel_name = binding.channel(device_name)
        try:
            padded = preprocess_mask_to_scan_grid(
                mask,
                total_x=total_x,
                total_y=total_y,
                scan_x_pixels=scan.x_pixels,
                extra_left=scan.extra_left,
                extra_right=scan.extra_right,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to preprocess mask for {channel_name}: {exc}") from exc
        if not np.any(padded):
            continue
        ttl = np.zeros((total_y, total_x, pixel_samples), dtype=bool)
        ttl[padded] = True
        ttl_signals[channel_name] = ttl.reshape(-1)

    return ttl_signals


# --------------------------------------------------------------------------- #
# The scan                                                                     #
# --------------------------------------------------------------------------- #


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
                extract_kept_samples(ch_data, total_y, total_x, samples_per_pixel, extra_left, x_pixels)
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
                try:
                    static_do_task.write([not v for v in static_values], auto_start=True)  # pyright:ignore
                except Exception:
                    pass
            static_do_task.close()


def raster_scan(
    *,
    daq: DAQ,
    galvo: Galvo,
    scan: ScanGroup,
    sample_rate_hz: float,
    ttl: dict[str, np.ndarray],
) -> np.ndarray:
    """Perform one confocal raster scan and return a ``(C, H, W)`` float32 frame.

    The devices supply their own wiring; the caller supplies the run's geometry.
    ``ttl`` has no default: an empty dict means "no mask drives a digital line",
    and the caller has to say so rather than leaving it off and finding out on
    the instrument which of the two it meant.
    """
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
    return reshape_to_frame(scan_data, total_y_out, x_out, px_out)


# --------------------------------------------------------------------------- #
# The program                                                                  #
# --------------------------------------------------------------------------- #


def channel_labels(daq: DAQ) -> list[str]:
    return [f"ai{index}" for index in daq.config.ai_channels]


def build_ttl(
    scan: ScanGroup, modulation: ModulationGroup, daq_params: DaqGroup, daq: DAQ
) -> dict:
    """Load the bound masks and turn them into per-pixel TTL waveforms.

    Done once before the loop rather than once per frame.
    """
    if not modulation.masks:
        return {}
    loaded = [(mask, mask.load()) for mask in modulation.masks]
    return mask_ttl(
        loaded,
        scan=scan,
        pixel_samples=pixel_samples(scan.dwell_time_us, daq_params.sample_rate_hz),
        device_name=daq.config.device_name,
    )


@program_registry.register("confocal")
class Confocal(Program):
    uses = [Galvo, DAQ]
    params = [ScanGroup, DaqGroup, ModulationGroup, FramesGroup]
    emits = {"intensity": Image2D}

    def run(self, ctx) -> None:
        scan = ctx.params[ScanGroup]
        daq_params = ctx.params[DaqGroup]
        modulation = ctx.params[ModulationGroup]
        num_frames = ctx.params[FramesGroup].num_frames

        daq: DAQ = ctx.devices[DAQ]
        galvo: Galvo = ctx.devices[Galvo]

        ttl = build_ttl(scan, modulation, daq_params, daq)
        labels = channel_labels(daq)
        total = "" if ctx.continuous else f"/{num_frames}"

        for index in ctx.frames(num_frames):
            ctx.status(f"frame {index + 1}{total}")
            frame = raster_scan(
                daq=daq,
                galvo=galvo,
                scan=scan,
                sample_rate_hz=daq_params.sample_rate_hz,
                ttl=ttl,
            )
            ctx.publish("intensity", frame, channels=labels)
