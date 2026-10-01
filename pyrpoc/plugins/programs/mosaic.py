"""Mosaic: a confocal frame at each stop of a stage grid, walked as a snake.

The waveform arithmetic, NI task setup and mask TTL below are copies of
``confocal.py``'s and must stay identical to them: each tile is a confocal
frame. What differs is at the bottom: the stage moves between frames, and the
grid is recorded with the output so a display can place each frame.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import nidaqmx as nx
import numpy as np
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader

from pyrpoc.plugins.devices import DAQ, DaqError, Galvo, PriorStage
from pyrpoc.structs.data_library.data import Image2D
from pyrpoc.structs.plugins.programs.program import (
    Cancelled,
    Program,
    RunContext,
    program_registry,
)

from .components.param_groups import (
    DaqGroup,
    Mask,
    ModulationGroup,
    MosaicGroup,
    ScanGroup,
    Tile,
)
from .components.runners import Single


def pixel_samples(dwell_time_us: float, sample_rate_hz: float) -> int:
    """Samples per pixel: truncating, floor 1. FLIM's rounds with floor 2,
    since its counter needs a high and a low tick; do not unify them."""
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


def raster_scan(
    *,
    daq: DAQ,
    galvo: Galvo,
    scan: ScanGroup,
    sample_rate_hz: float,
    ttl: dict[str, np.ndarray],
) -> np.ndarray:
    """One confocal raster scan, as a ``(C, H, W)`` float32 frame."""
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


def channel_labels(daq: DAQ) -> list[str]:
    return [f"ai{index}" for index in daq.config.ai_channels]


def build_ttl(
    scan: ScanGroup, modulation: ModulationGroup, daq_params: DaqGroup, daq: DAQ
) -> dict[str, np.ndarray]:
    """The bound masks as per-pixel TTL waveforms, built once before the loop."""
    return mask_ttl(
        modulation.masks,
        scan=scan,
        pixel_samples=pixel_samples(scan.dwell_time_us, daq_params.sample_rate_hz),
        device_name=daq.config.device_name,
    )


def move_stage(ctx: RunContext, stage: PriorStage, x_um: float, y_um: float) -> None:
    """Move and wait for the stage to stop; a stopped run interrupts the wait."""
    stage.move_xy_to(x_um, y_um)
    stage.wait_until_stopped("xy", ctx.sleep)


@program_registry.register("mosaic")
class Mosaic(Program):
    display_name = "Mosaic"
    uses = [Galvo, DAQ, PriorStage]
    params = [ScanGroup, DaqGroup, ModulationGroup, MosaicGroup]
    emits = {"intensity": Image2D}
    # Once only: a mosaic is one pass over the grid, never repeated or looped.
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        grid = ctx.params[MosaicGroup]
        stage = ctx.devices[PriorStage]
        origin = stage.xy_position()
        tiles = grid.snake(origin)
        # Before the first tile, so a recording cut short still says where its tiles are.
        ctx.describe("intensity", mosaic=grid.layout_metadata(tiles))

        try:
            self.image_tiles(ctx, stage, tiles)
        except Cancelled:
            stage.stop_smoothly()
            raise
        # Back where it started, so a repeat run images the same area.
        ctx.status("returning to the start position")
        move_stage(ctx, stage, *origin)

    @staticmethod
    def image_tiles(ctx: RunContext, stage: PriorStage, tiles: list[Tile]) -> None:
        scan = ctx.params[ScanGroup]
        daq_params = ctx.params[DaqGroup]
        daq = ctx.devices[DAQ]
        galvo = ctx.devices[Galvo]
        ttl = build_ttl(scan, ctx.params[ModulationGroup], daq_params, daq)
        labels = channel_labels(daq)

        for tile in tiles:
            ctx.check_cancel()
            ctx.status(
                f"tile {tile.index + 1}/{len(tiles)} (row {tile.row + 1}, col {tile.col + 1})"
            )
            move_stage(ctx, stage, tile.x_um, tile.y_um)
            frame = raster_scan(
                daq=daq,
                galvo=galvo,
                scan=scan,
                sample_rate_hz=daq_params.sample_rate_hz,
                ttl=ttl,
            )
            ctx.publish("intensity", frame, channels=labels)
