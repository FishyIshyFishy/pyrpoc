"""The galvo raster: its waveform, and one run's scan built from it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.core.program import RunContext
from pyrpoc.plugins.devices import DAQ, Galvo
from pyrpoc.plugins.program_io.inputs import (
    MASK_LINES,
    MASKS,
    ScanGeometry,
    lines_for,
    rate_hz,
)

from .daq_tasks import acquire, galvo_channels
from .mask_ttl import mask_ttl


def pixel_samples(dwell_time_us: float, sample_rate_hz: float) -> int:
    """Samples per pixel, truncated, at least 1."""
    # FLIM's rounds with a floor of 2, since its counter needs a high and a low tick; do not unify
    # them.
    return max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))


def raster_waveform(scan: ScanGeometry, samples_per_pixel: int) -> np.ndarray:
    """The ``(fast, slow)`` AO waveform for one frame: the fast axis sweeps the
    overscan-padded width each line, holding ``samples_per_pixel`` per pixel."""
    fast_axis = scan.fast_volts(np.arange(-scan.extra_left, scan.x_pixels + scan.extra_right))
    slow_axis = scan.slow_volts(np.arange(scan.y_pixels))
    fast_raster = np.tile(np.repeat(fast_axis, samples_per_pixel), scan.y_pixels)
    slow_raster = np.repeat(slow_axis, scan.total_x * samples_per_pixel)
    return np.vstack((fast_raster, slow_raster)).astype(np.float64)


@dataclass(frozen=True)
class Drive:
    """One galvo a scan moves, and the volts added to the scan's on each axis to point it at
    the same spot."""

    galvo: Galvo
    fast_offset_v: float
    slow_offset_v: float

    def rows(self, waveform: np.ndarray) -> np.ndarray:
        """The scan's ``(fast, slow)`` ``waveform`` for this galvo."""
        return waveform + np.array([[self.fast_offset_v], [self.slow_offset_v]])


@dataclass(frozen=True)
class Raster:
    """One run's scan: its waveform, one ``(fast, slow)`` pair of rows per galvo it drives,
    its mask TTL and channels."""

    # The waveform and mask TTL are the same every frame, so they are built once, before the
    # first.

    daq: DAQ
    ao_channels: list[int]
    scan: ScanGeometry
    sample_rate_hz: float
    samples_per_pixel: int
    waveform: np.ndarray
    ttl: dict[str, np.ndarray]

    @classmethod
    def for_run(cls, ctx: RunContext) -> Raster:
        """The scan on the program's one galvo."""
        return cls.driving(ctx, [Drive(ctx.devices[Galvo], 0.0, 0.0)])

    @classmethod
    def driving(cls, ctx: RunContext, drives: list[Drive]) -> Raster:
        """The scan on every galvo of ``drives``, all on one clock."""
        scan = ScanGeometry.from_inputs(ctx.inputs)
        sample_rate_hz = rate_hz(ctx.inputs)
        daq = ctx.devices[DAQ]
        samples_per_pixel = pixel_samples(scan.dwell_time_us, sample_rate_hz)
        masks = ctx.inputs[MASKS]
        lines = lines_for(masks, ctx.inputs[MASK_LINES])
        ttl = mask_ttl(masks, lines, scan, samples_per_pixel, daq.config.device_name)
        waveform = raster_waveform(scan, samples_per_pixel)
        return cls(
            daq,
            [channel for drive in drives for channel in galvo_channels(drive.galvo)],
            scan,
            sample_rate_hz,
            samples_per_pixel,
            np.vstack([drive.rows(waveform) for drive in drives]),
            ttl,
        )

    @property
    def channel_labels(self) -> list[str]:
        return [f"ai{index}" for index in self.daq.config.ai_channels]

    def samples(self) -> np.ndarray:
        """One scan as ``(C, H, W, S)``: every displayed pixel's raw samples,
        with the overscan columns dropped."""
        raw = acquire(self.daq, self.ao_channels, self.waveform, self.ttl, self.sample_rate_hz)
        scan = self.scan
        cube = raw.astype(np.float32).reshape(
            len(raw), scan.y_pixels, scan.total_x, self.samples_per_pixel
        )
        return np.ascontiguousarray(cube[:, :, scan.kept_columns, :])

    def frame(self) -> np.ndarray:
        """One scan as a ``(C, H, W)`` frame, each pixel the mean of its samples."""
        return self.samples().mean(axis=3)
