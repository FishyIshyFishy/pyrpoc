"""The galvo raster: its waveform, and one run's scan built from it. Confocal,
split confocal and the mosaic play the scan; FLIM plays the waveform with its
own sample count."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.plugins.devices import DAQ, Galvo
from pyrpoc.structs.plugins.programs.program import RunContext

from ..parameter_groups.daq import DaqGroup
from ..parameter_groups.modulation import ModulationGroup
from ..parameter_groups.scan import ScanGroup
from .daq_tasks import acquire
from .mask_ttl import mask_ttl


def pixel_samples(dwell_time_us: float, sample_rate_hz: float) -> int:
    """Samples per pixel: truncating, floor 1. FLIM's rounds with floor 2,
    since its counter needs a high and a low tick; do not unify them."""
    return max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))


def raster_waveform(scan: ScanGroup, samples_per_pixel: int) -> np.ndarray:
    """The ``(fast, slow)`` AO waveform for one frame: the fast axis sweeps the
    overscan-padded width each line, holding ``samples_per_pixel`` per pixel."""
    fast_step = 2.0 * scan.fast_axis_amplitude / scan.x_pixels
    fast_start = -scan.fast_axis_amplitude - scan.extra_left * fast_step
    fast_axis = (
        fast_start + np.arange(scan.total_x, dtype=np.float32) * fast_step + scan.fast_axis_offset
    )
    slow_axis = (
        np.linspace(-1.0, 1.0, scan.y_pixels, endpoint=False, dtype=np.float32)
        * scan.slow_axis_amplitude
        + scan.slow_axis_offset
    )
    fast_raster = np.tile(np.repeat(fast_axis, samples_per_pixel), scan.y_pixels)
    slow_raster = np.repeat(slow_axis, scan.total_x * samples_per_pixel)
    return np.vstack((fast_raster, slow_raster)).astype(np.float64)


@dataclass(frozen=True)
class Raster:
    """One run's scan. The waveform and mask TTL are the same every frame, so
    they are built once, before the first."""

    daq: DAQ
    galvo: Galvo
    scan: ScanGroup
    sample_rate_hz: float
    samples_per_pixel: int
    waveform: np.ndarray
    ttl: dict[str, np.ndarray]

    @classmethod
    def for_run(cls, ctx: RunContext) -> Raster:
        scan = ctx.params[ScanGroup]
        sample_rate_hz = ctx.params[DaqGroup].sample_rate_hz
        daq = ctx.devices[DAQ]
        samples_per_pixel = pixel_samples(scan.dwell_time_us, sample_rate_hz)
        ttl = mask_ttl(
            ctx.params[ModulationGroup].masks, scan, samples_per_pixel, daq.config.device_name
        )
        waveform = raster_waveform(scan, samples_per_pixel)
        return cls(daq, ctx.devices[Galvo], scan, sample_rate_hz, samples_per_pixel, waveform, ttl)

    @property
    def channel_labels(self) -> list[str]:
        return [f"ai{index}" for index in self.daq.config.ai_channels]

    def samples(self) -> np.ndarray:
        """One scan as ``(C, H, W, S)``: every displayed pixel's raw samples,
        with the overscan columns dropped."""
        raw = acquire(self.daq, self.galvo, self.waveform, self.ttl, self.sample_rate_hz)
        scan = self.scan
        cube = raw.astype(np.float32).reshape(
            len(raw), scan.y_pixels, scan.total_x, self.samples_per_pixel
        )
        return np.ascontiguousarray(cube[:, :, scan.kept_columns, :])

    def frame(self) -> np.ndarray:
        """One scan as a ``(C, H, W)`` frame, each pixel the mean of its samples."""
        return self.samples().mean(axis=3)
